"""频域局部对比分析 GUI。

用法：python frequency_analysis/frequency_gui.py
- 工具栏“打开图片”可多选任意数量图片；“打开文件夹”可导入整个目录的图片
  （导入时弹出选择窗口：可勾选/筛选/预览缩略图/限制加载数量，避免一次导入过多导致崩溃）
- 在左侧图片上用鼠标拖拽框选任意区域，松开鼠标即生成该区域的频谱图
- 可连续框选，任意数量；频谱结果面板为全局视图：所有图片的框选结果集中显示、不随图片切换清空
- 左/右方向键或工具栏切换图片（仅切换左侧画布与 ROI，右侧结果保留）
- 底部“图像增强”面板可调亮度/对比度/伽马/锐化/CLAHE/反色，
  所见即所分析：调整同时作用于显示与频谱计算；每张图独立参数
- 工具栏“导出”将所有频谱图、空间patch导出为PNG，框选坐标+增强参数导出为CSV
"""

import csv
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal, QTimer
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


MIN_ROI = 8
CARD_IMG_H = 110
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def compute_spectrum(patch: np.ndarray) -> np.ndarray:
    if patch.shape[0] < 32 or patch.shape[1] < 32:
        patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_LINEAR)
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    return 20.0 * np.log(np.abs(fshift) + 1.0)


def analyze_region(image_bgr: np.ndarray, x: int, y: int, w: int, h: int):
    patch = cv2.cvtColor(image_bgr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    mag = compute_spectrum(patch)
    mn, mx = float(mag.min()), float(mag.max())
    norm = ((mag - mn) / (mx - mn + 1e-9) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return patch, mag, rgb


def gray_to_qimage(gray: np.ndarray) -> QImage:
    h, w = gray.shape
    return QImage(gray.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    h, w, _ = rgb.shape
    return QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


@dataclass
class AdjustParams:
    brightness: int = 0       # -100..100
    contrast: int = 0         # -100..100
    gamma: float = 1.0        # 0.1..3.0
    sharpen: int = 0          # 0..100
    clahe: float = 0.0        # 0..10, 0=off
    invert: bool = False

    def is_default(self) -> bool:
        return (
            self.brightness == 0
            and self.contrast == 0
            and abs(self.gamma - 1.0) < 1e-3
            and self.sharpen == 0
            and self.clahe == 0.0
            and not self.invert
        )

    def signature(self):
        return (
            self.brightness,
            self.contrast,
            round(self.gamma, 3),
            self.sharpen,
            round(self.clahe, 2),
            self.invert,
        )

    def copy(self) -> "AdjustParams":
        return AdjustParams(
            self.brightness, self.contrast, self.gamma,
            self.sharpen, self.clahe, self.invert,
        )


def apply_adjustments(img_bgr: np.ndarray, p: AdjustParams) -> np.ndarray:
    if p.is_default():
        return img_bgr
    out = img_bgr.astype(np.float32)
    # 1) 亮度 + 对比度（点运算，围绕 128 拉伸）
    if p.brightness != 0 or p.contrast != 0:
        factor = 1.0 + p.contrast / 100.0
        out = (out - 128.0) * factor + 128.0 + p.brightness
        out = np.clip(out, 0, 255)
    # 2) 伽马（非线性映射，提亮/压暗暗部）
    if abs(p.gamma - 1.0) > 1e-3:
        lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / p.gamma) * 255.0
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        out = lut[np.clip(out, 0, 255).astype(np.uint8)].astype(np.float32)
    work = np.clip(out, 0, 255).astype(np.uint8)
    # 3) CLAHE 自适应局部均衡（在 LAB 的 L 通道上做）
    if p.clahe > 0:
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=float(p.clahe), tileGridSize=(8, 8))
        l = cl.apply(l)
        work = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    out = work.astype(np.float32)
    # 4) 锐化（Unsharp Masking，强化边缘高频）
    if p.sharpen > 0:
        amount = p.sharpen / 100.0
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
        out = out + amount * (out - blur)
    # 5) 反色
    if p.invert:
        out = 255.0 - out
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class Region:
    idx: int
    x: int
    y: int
    w: int
    h: int
    patch_gray: np.ndarray
    spectrum_mag: np.ndarray
    spectrum_rgb: np.ndarray


class ImageSession:
    _sid_counter = 0

    def __init__(self, path: str):
        ImageSession._sid_counter += 1
        self.sid = ImageSession._sid_counter
        self.path = path
        # 懒加载：仅在显示/分析时读取原图，避免一次导入大量图片时内存爆炸
        self._image_bgr: np.ndarray | None = None
        self._image_failed = False
        self.regions: list[Region] = []
        self.next_idx = 1
        self.adjust = AdjustParams()
        self._adj_cache: np.ndarray | None = None
        self._adj_sig = None

    @property
    def image_bgr(self) -> np.ndarray:
        if self._image_failed:
            raise ValueError(f"无法读取图像: {self.path}")
        if self._image_bgr is None:
            img = cv2.imread(self.path)
            if img is None:
                self._image_failed = True
                raise ValueError(f"无法读取图像: {self.path}")
            self._image_bgr = img
        return self._image_bgr

    def release_image(self):
        """释放原图与增强缓存以节省内存；已计算的频谱结果保留。"""
        self._image_bgr = None
        self._adj_cache = None
        self._adj_sig = None

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    def adjusted_bgr(self) -> np.ndarray:
        sig = self.adjust.signature()
        if sig != self._adj_sig or self._adj_cache is None:
            self._adj_cache = apply_adjustments(self.image_bgr, self.adjust)
            self._adj_sig = sig
        return self._adj_cache

    def add_region(self, x: int, y: int, w: int, h: int) -> Region:
        adj = self.adjusted_bgr()
        patch, mag, rgb = analyze_region(adj, x, y, w, h)
        region = Region(self.next_idx, x, y, w, h, patch, mag, rgb)
        self.next_idx += 1
        self.regions.append(region)
        return region

    def recompute_regions(self):
        adj = self.adjusted_bgr()
        for r in self.regions:
            patch, mag, rgb = analyze_region(adj, r.x, r.y, r.w, r.h)
            r.patch_gray, r.spectrum_mag, r.spectrum_rgb = patch, mag, rgb

    def remove_region(self, idx: int) -> Region | None:
        for i, r in enumerate(self.regions):
            if r.idx == idx:
                return self.regions.pop(i)
        return None


class RoiItem(QGraphicsRectItem):
    def __init__(self, idx: int, x: int, y: int, w: int, h: int):
        super().__init__(x, y, w, h)
        self.idx = idx
        self.setPen(QPen(QColor("#00ff66"), 2))
        self.setZValue(2)
        txt = QGraphicsTextItem(str(idx), self)
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        txt.setFont(f)
        txt.setDefaultTextColor(QColor("black"))
        txt.setZValue(4)
        br = txt.boundingRect()
        bg = QGraphicsRectItem(self)
        bg.setRect(x + 2, y - br.height() - 1, br.width() + 6, br.height() + 1)
        bg.setBrush(QColor("#00ff66"))
        bg.setPen(QPen(Qt.NoPen))
        bg.setZValue(3)
        txt.setPos(x + 5, y - br.height() - 1)


class ImageCanvas(QGraphicsView):
    regionCreated = Signal(int, int, int, int)

    def __init__(self):
        super().__init__()
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_rect = QRectF()
        self._rubber: QGraphicsRectItem | None = None
        self._origin: QPointF | None = None

    def set_image(self, qimg: QImage):
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(QPixmap.fromImage(qimg))
        self._image_rect = QRectF(0, 0, qimg.width(), qimg.height())
        self._scene.setSceneRect(self._image_rect)
        self.resetTransform()
        self.fitInView(self._image_rect, Qt.KeepAspectRatio)

    def update_pixmap(self, qimg: QImage):
        """仅替换显示像素，保留缩放/平移与已有 ROI（尺寸不变时使用）。"""
        if self._pixmap_item is None:
            self.set_image(qimg)
            return
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))

    def add_roi(self, region: Region) -> RoiItem:
        item = RoiItem(region.idx, region.x, region.y, region.w, region.h)
        self._scene.addItem(item)
        return item

    def remove_roi(self, item: RoiItem):
        self._scene.removeItem(item)

    def clear_rois(self):
        for it in list(self._scene.items()):
            if isinstance(it, RoiItem):
                self._scene.removeItem(it)

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._pixmap_item is not None:
            sp = self.mapToScene(e.position().toPoint())
            self._origin = sp
            self._rubber = QGraphicsRectItem()
            self._rubber.setPen(QPen(QColor("yellow"), 1, Qt.DashLine))
            self._rubber.setZValue(5)
            self._scene.addItem(self._rubber)
            self._rubber.setRect(QRectF(sp, sp))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._origin is not None and self._rubber is not None:
            sp = self.mapToScene(e.position().toPoint())
            self._rubber.setRect(QRectF(self._origin, sp).normalized())
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._origin is not None:
            sp = self.mapToScene(e.position().toPoint())
            rect = QRectF(self._origin, sp).normalized().intersected(self._image_rect)
            if self._rubber is not None:
                self._scene.removeItem(self._rubber)
                self._rubber = None
            self._origin = None
            if rect.width() >= MIN_ROI and rect.height() >= MIN_ROI:
                self.regionCreated.emit(
                    int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
                )
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e):
        if self._pixmap_item is not None:
            factor = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.scale(factor, factor)


class RegionCard(QFrame):
    removeRequested = Signal(int, int)  # sid, region idx

    def __init__(self, session: "ImageSession", region: Region):
        super().__init__()
        self.sid = session.sid
        self.idx = region.idx
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        self.patch_lbl = QLabel()
        self.patch_lbl.setPixmap(self._patch_pixmap(region))
        self.patch_lbl.setAlignment(Qt.AlignCenter)

        self.spec_lbl = QLabel()
        self.spec_lbl.setPixmap(self._spec_pixmap(region))
        self.spec_lbl.setAlignment(Qt.AlignCenter)

        info = QLabel(
            f"{session.stem}\n#{region.idx}\n{region.w}x{region.h}\n@({region.x},{region.y})"
        )
        info.setStyleSheet("color:#9ad;")
        info.setWordWrap(True)

        btn = QPushButton("删除")
        btn.clicked.connect(lambda: self.removeRequested.emit(self.sid, self.idx))

        lay.addWidget(self.patch_lbl)
        lay.addWidget(self.spec_lbl)
        lay.addWidget(info, 0, Qt.AlignVCenter)
        lay.addWidget(btn, 0, Qt.AlignVCenter)

    @staticmethod
    def _patch_pixmap(region: Region) -> QPixmap:
        return QPixmap.fromImage(gray_to_qimage(region.patch_gray)).scaledToHeight(
            CARD_IMG_H, Qt.SmoothTransformation
        )

    @staticmethod
    def _spec_pixmap(region: Region) -> QPixmap:
        return QPixmap.fromImage(rgb_to_qimage(region.spectrum_rgb)).scaledToHeight(
            CARD_IMG_H, Qt.SmoothTransformation
        )

    def update_pixmaps(self, region: Region):
        self.patch_lbl.setPixmap(self._patch_pixmap(region))
        self.spec_lbl.setPixmap(self._spec_pixmap(region))


class AdjustPanel(QWidget):
    changed = Signal()
    applyAllRequested = Signal()

    def __init__(self):
        super().__init__()
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)

        # 伽马/CLAHE 用整数滑块乘系数表示小数：gamma *100，clahe *2
        self.bri = self._mk_slider(-100, 100, 1, 0)
        self.con = self._mk_slider(-100, 100, 1, 0)
        self.gam = self._mk_slider(10, 300, 5, 100)
        self.shp = self._mk_slider(0, 100, 1, 0)
        self.cla = self._mk_slider(0, 20, 1, 0)

        self.bri_v = QLabel("0")
        self.con_v = QLabel("0")
        self.gam_v = QLabel("1.00")
        self.shp_v = QLabel("0")
        self.cla_v = QLabel("0.0")

        outer.addWidget(self._group("亮度", self.bri, self.bri_v))
        outer.addWidget(self._group("对比度", self.con, self.con_v))
        outer.addWidget(self._group("伽马", self.gam, self.gam_v))
        outer.addWidget(self._group("锐化", self.shp, self.shp_v))
        outer.addWidget(self._group("CLAHE", self.cla, self.cla_v))

        for s in (self.bri, self.con, self.gam, self.shp, self.cla):
            s.valueChanged.connect(self._on_slider)

        self.inv = QCheckBox("反色")
        self.inv.toggled.connect(lambda _: self.changed.emit())
        outer.addWidget(self.inv, 0, Qt.AlignTop)

        actions = QVBoxLayout()
        actions.setSpacing(4)
        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self.reset)
        btn_all = QPushButton("应用到全部")
        btn_all.clicked.connect(self.applyAllRequested.emit)
        actions.addWidget(btn_reset)
        actions.addWidget(btn_all)
        actions.addStretch(1)
        outer.addLayout(actions)
        outer.addStretch(1)

    @staticmethod
    def _mk_slider(lo: int, hi: int, step: int, val: int) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setValue(val)
        s.setMinimumWidth(130)
        return s

    @staticmethod
    def _group(name: str, slider: QSlider, value_lbl: QLabel) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 0, 6, 0)
        lay.setSpacing(2)
        head = QHBoxLayout()
        nm = QLabel(name)
        nm.setStyleSheet("font-weight:bold;")
        head.addWidget(nm)
        head.addStretch(1)
        head.addWidget(value_lbl)
        lay.addLayout(head)
        lay.addWidget(slider)
        return w

    def _on_slider(self):
        self.bri_v.setText(str(self.bri.value()))
        self.con_v.setText(str(self.con.value()))
        self.gam_v.setText(f"{self.gam.value() / 100.0:.2f}")
        self.shp_v.setText(str(self.shp.value()))
        self.cla_v.setText(f"{self.cla.value() / 2.0:.1f}")
        self.changed.emit()

    def params(self) -> AdjustParams:
        return AdjustParams(
            brightness=self.bri.value(),
            contrast=self.con.value(),
            gamma=self.gam.value() / 100.0,
            sharpen=self.shp.value(),
            clahe=self.cla.value() / 2.0,
            invert=self.inv.isChecked(),
        )

    def set_params(self, p: AdjustParams):
        self.bri.setValue(p.brightness)
        self.con.setValue(p.contrast)
        self.gam.setValue(int(round(p.gamma * 100)))
        self.shp.setValue(p.sharpen)
        self.cla.setValue(int(round(p.clahe * 2)))
        self.inv.setChecked(p.invert)
        self._refresh_labels()

    def _refresh_labels(self):
        self.bri_v.setText(str(self.bri.value()))
        self.con_v.setText(str(self.con.value()))
        self.gam_v.setText(f"{self.gam.value() / 100.0:.2f}")
        self.shp_v.setText(str(self.shp.value()))
        self.cla_v.setText(f"{self.cla.value() / 2.0:.1f}")

    def reset(self):
        self.blockSignals(True)
        self.set_params(AdjustParams())
        self.blockSignals(False)
        self.changed.emit()


class FolderImportDialog(QDialog):
    """打开文件夹时弹出：列出所有图片文件、可勾选/筛选/预览/限制数量。"""

    SUGGEST_LIMIT = 100
    WARN_LIMIT = 500
    PREVIEW_H = 200
    THUMB_CACHE_MAX = 32

    def __init__(self, folder: str, paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("导入文件夹图片")
        self.resize(760, 620)
        self._paths = paths
        self._thumb_cache: dict[str, QPixmap] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        info = QLabel(f"文件夹：{folder}\n共找到 {len(paths)} 张图片（已按文件名排序）")
        info.setStyleSheet("color:#9ad;")
        outer.addWidget(info)

        filt = QHBoxLayout()
        filt.addWidget(QLabel("筛选："))
        self.filt_edit = QLineEdit()
        self.filt_edit.setPlaceholderText("按文件名筛选（不区分大小写）")
        self.filt_edit.textChanged.connect(self._apply_filter)
        filt.addWidget(self.filt_edit, 1)
        outer.addLayout(filt)

        self.list = QListWidget()
        self.list.setAlternatingRowColors(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.itemSelectionChanged.connect(self._on_select)
        self.list.itemChanged.connect(self._on_item_changed)
        for p in paths:
            it = QListWidgetItem(os.path.basename(p))
            it.setData(Qt.UserRole, p)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.list.addItem(it)
        outer.addWidget(self.list, 1)

        self.preview = QLabel("选择文件以预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(self.PREVIEW_H)
        self.preview.setStyleSheet("background:#222; color:#888; border:1px solid #333;")
        outer.addWidget(self.preview)

        row = QHBoxLayout()
        b_all = QPushButton("全选")
        b_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        b_none = QPushButton("全不选")
        b_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        b_inv = QPushButton("反选")
        b_inv.clicked.connect(self._invert)
        row.addWidget(b_all)
        row.addWidget(b_none)
        row.addWidget(b_inv)
        row.addStretch(1)
        self.count_lbl = QLabel()
        row.addWidget(self.count_lbl)
        outer.addLayout(row)

        lim = QHBoxLayout()
        lim.addWidget(QLabel("最多加载："))
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(1, max(1, len(paths)))
        self.limit_spin.setValue(min(len(paths), self.SUGGEST_LIMIT))
        self.limit_spin.valueChanged.connect(self._update_warning)
        lim.addWidget(self.limit_spin)
        lim.addWidget(QLabel(f"张    （建议 ≤ {self.SUGGEST_LIMIT}，超过 {self.WARN_LIMIT} 可能较慢）"))
        lim.addStretch(1)
        outer.addLayout(lim)

        self.warn_lbl = QLabel("")
        self.warn_lbl.setStyleSheet("color:#e8a;")
        outer.addWidget(self.warn_lbl)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("加载")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        self._update_count()
        self._update_warning()

    def _apply_filter(self, text: str):
        key = text.strip().lower()
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setHidden(bool(key) and key not in it.text().lower())

    def _set_all(self, state):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it.isHidden():
                it.setCheckState(state)
        self.list.blockSignals(False)
        self._update_count()
        self._update_warning()

    def _invert(self):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it.isHidden():
                it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked else Qt.Checked)
        self.list.blockSignals(False)
        self._update_count()
        self._update_warning()

    def _on_item_changed(self, *_):
        self._update_count()
        self._update_warning()

    def _checked_paths(self) -> list[str]:
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if not it.isHidden() and it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole))
        return out

    def _update_count(self):
        checked = self._checked_paths()
        visible = sum(1 for i in range(self.list.count()) if not self.list.item(i).isHidden())
        self.count_lbl.setText(f"已选 {len(checked)} / 显示 {visible}")

    def _update_warning(self):
        checked = len(self._checked_paths())
        limit = self.limit_spin.value()
        if checked > limit:
            self.warn_lbl.setText(f"将只加载前 {limit} 张（当前已选 {checked} 张）")
        elif limit > self.WARN_LIMIT:
            self.warn_lbl.setText(f"将加载 {limit} 张，数量较多可能较慢（内存按需加载）")
        else:
            self.warn_lbl.setText("")

    def _on_select(self):
        it = self.list.currentItem()
        if it is None:
            self.preview.setText("选择文件以预览")
            return
        pm = self._load_thumb(it.data(Qt.UserRole))
        if pm is None:
            self.preview.setText("无法预览")
        else:
            self.preview.setPixmap(pm)

    def _load_thumb(self, path: str) -> QPixmap | None:
        if path in self._thumb_cache:
            return self._thumb_cache[path]
        # 用 1/8 缩略图模式读取，降低预览内存与耗时；失败则回退到正常读取
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
        if img is None:
            img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = self.PREVIEW_H / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pm = QPixmap.fromImage(rgb_to_qimage(rgb))
        if len(self._thumb_cache) > self.THUMB_CACHE_MAX:
            self._thumb_cache.pop(next(iter(self._thumb_cache)))
        self._thumb_cache[path] = pm
        return pm

    def selected_paths(self) -> list[str]:
        return self._checked_paths()[: self.limit_spin.value()]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("频域局部对比分析")
        self.resize(1500, 960)

        self.sessions: list[ImageSession] = []
        self.current: int = -1
        self._roi_items: dict[int, RoiItem] = {}
        # 全局结果面板：卡片按 (sid, region_idx) 索引，跨所有图片常驻显示
        self._cards: dict[tuple[int, int], RegionCard] = {}

        self._build_toolbar()
        self._build_central()
        self._build_adjust_dock()
        self.setStatusBar(QStatusBar())

        self._adj_timer = QTimer(self)
        self._adj_timer.setSingleShot(True)
        self._adj_timer.setInterval(30)
        self._adj_timer.timeout.connect(self._apply_adjust)

        self._update_actions()
        self._update_status()

    def _build_toolbar(self):
        tb = QToolBar("主工具栏", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        self.toolbar = tb

        self.act_open = QAction("打开图片", self)
        self.act_open.setShortcut(QKeySequence.Open)
        self.act_open.triggered.connect(self.open_images)
        tb.addAction(self.act_open)

        self.act_open_folder = QAction("打开文件夹", self)
        self.act_open_folder.triggered.connect(self.open_folder)
        tb.addAction(self.act_open_folder)

        self.act_prev = QAction("上一张", self)
        self.act_prev.setShortcut(QKeySequence(Qt.Key_Left))
        self.act_prev.triggered.connect(self.prev_image)
        tb.addAction(self.act_prev)

        self.act_next = QAction("下一张", self)
        self.act_next.setShortcut(QKeySequence(Qt.Key_Right))
        self.act_next.triggered.connect(self.next_image)
        tb.addAction(self.act_next)

        tb.addSeparator()

        self.act_clear = QAction("清空当前图", self)
        self.act_clear.triggered.connect(self.clear_current)
        tb.addAction(self.act_clear)

        self.act_clear_all = QAction("清空全部结果", self)
        self.act_clear_all.triggered.connect(self.clear_all_results)
        tb.addAction(self.act_clear_all)

        self.act_export = QAction("导出结果", self)
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.triggered.connect(self.export_results)
        tb.addAction(self.act_export)

    def _build_adjust_dock(self):
        self.panel = AdjustPanel()
        self.panel.changed.connect(self._schedule_adjust)
        self.panel.applyAllRequested.connect(self.apply_adjust_to_all)

        dock = QDockWidget("图像增强（所见即所分析：调整同时作用于显示与频谱）", self)
        dock.setWidget(self.panel)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.dock = dock

        self.toolbar.addSeparator()
        self.act_adjust = QAction("图像增强", self)
        self.act_adjust.setCheckable(True)
        self.act_adjust.setChecked(True)
        self.act_adjust.triggered.connect(dock.setVisible)
        self.toolbar.addAction(self.act_adjust)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        self.image_title = QLabel("请打开图片")
        self.image_title.setStyleSheet("padding:6px; color:#ddd; background:#222;")
        left_lay.addWidget(self.image_title)
        self.canvas = ImageCanvas()
        self.canvas.regionCreated.connect(self.on_region_created)
        left_lay.addWidget(self.canvas, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(6, 6, 6, 6)
        right_title = QLabel("全部频谱结果（跨所有图片，按框选顺序）")
        right_title.setStyleSheet("font-weight:bold; padding:4px;")
        right_lay.addWidget(right_title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(6)
        self.cards_layout.addStretch(1)
        self.scroll.setWidget(self.cards_container)
        right_lay.addWidget(self.scroll, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    # ---- 加载图片 ----
    def _load_paths(self, paths: list[str]):
        loaded = []
        for p in paths:
            if not os.path.isfile(p):
                continue
            self.sessions.append(ImageSession(p))
            loaded.append(p)
        if loaded:
            self.current = len(self.sessions) - len(loaded)
            self.show_current()
        self._update_actions()
        self._update_status()

    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        self._load_paths(paths)

    def open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not d:
            return
        paths = [
            os.path.join(d, fn)
            for fn in sorted(os.listdir(d))
            if fn.lower().endswith(IMAGE_EXTS) and os.path.isfile(os.path.join(d, fn))
        ]
        if not paths:
            QMessageBox.information(self, "提示", "该文件夹下未找到支持的图片。")
            return
        dlg = FolderImportDialog(d, paths, self)
        if dlg.exec() != QDialog.Accepted:
            return
        selected = dlg.selected_paths()
        if not selected:
            return
        self._load_paths(selected)

    # ---- 显示 ----
    def show_current(self):
        while True:
            s = self._current_session()
            if s is None:
                self.canvas.set_image(self._empty_qimage())
                self._rebuild_rois()
                self._load_params_to_panel()
                self._update_image_title()
                self._update_status()
                return
            try:
                qimg = self._current_qimage()
            except ValueError as ex:
                QMessageBox.warning(self, "读取失败", str(ex))
                self.sessions.pop(self.current)
                if self.current >= len(self.sessions):
                    self.current = len(self.sessions) - 1
                if self.current < 0:
                    self.current = -1
                continue
            self.canvas.set_image(qimg)
            self._rebuild_rois()
            # 结果面板为全局视图：切换图片时不清空卡片，仅同步当前图增强参数
            self._load_params_to_panel()
            self._update_image_title()
            self._release_inactive_images()
            self._update_status()
            return

    def _current_session(self) -> ImageSession | None:
        if 0 <= self.current < len(self.sessions):
            return self.sessions[self.current]
        return None

    def _session_by_sid(self, sid: int) -> ImageSession | None:
        for s in self.sessions:
            if s.sid == sid:
                return s
        return None

    def _empty_qimage(self) -> QImage:
        img = QImage(2, 2, QImage.Format_RGB888)
        img.fill(QColor(30, 30, 30))
        return img

    def _release_inactive_images(self):
        """释放非当前图片的原图与增强缓存，限制内存占用（频谱结果保留）。"""
        cur = self._current_session()
        for s in self.sessions:
            if s is not cur:
                s.release_image()

    def _current_qimage(self) -> QImage:
        s = self._current_session()
        rgb = cv2.cvtColor(s.adjusted_bgr(), cv2.COLOR_BGR2RGB)
        return rgb_to_qimage(rgb)

    def _rebuild_rois(self):
        self._roi_items.clear()
        self.canvas.clear_rois()
        s = self._current_session()
        if s is None:
            return
        for r in s.regions:
            self._roi_items[r.idx] = self.canvas.add_roi(r)

    def _add_card(self, session: ImageSession, region: Region):
        card = RegionCard(session, region)
        card.removeRequested.connect(self.on_remove_region)
        self._cards[(session.sid, region.idx)] = card
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)

    def _remove_card(self, sid: int, idx: int):
        card = self._cards.pop((sid, idx), None)
        if card is not None:
            self.cards_layout.removeWidget(card)
            card.setParent(None)

    def _load_params_to_panel(self):
        s = self._current_session()
        if s is None:
            return
        self.panel.blockSignals(True)
        self.panel.set_params(s.adjust)
        self.panel.blockSignals(False)

    # ---- 增强处理 ----
    def _schedule_adjust(self):
        s = self._current_session()
        if s is None:
            return
        s.adjust = self.panel.params()
        self._adj_timer.start()

    def _apply_adjust(self):
        s = self._current_session()
        if s is None:
            return
        self.canvas.update_pixmap(self._current_qimage())
        s.recompute_regions()
        for r in s.regions:
            card = self._cards.get((s.sid, r.idx))
            if card is not None:
                card.update_pixmaps(r)
        self._update_status()

    def apply_adjust_to_all(self):
        src = self.panel.params()
        cur = self._current_session()
        for s in self.sessions:
            s.adjust = src.copy()
            try:
                s.recompute_regions()
            except ValueError:
                if s is not cur:
                    s.release_image()
                continue
            for r in s.regions:
                card = self._cards.get((s.sid, r.idx))
                if card is not None:
                    card.update_pixmaps(r)
            # 非当前图片处理完即释放原图，避免一次性占用过多内存
            if s is not cur:
                s.release_image()
        if cur is not None:
            try:
                self.canvas.update_pixmap(self._current_qimage())
            except ValueError:
                pass
        self._update_status()

    # ---- 区域增删 ----
    def on_region_created(self, x: int, y: int, w: int, h: int):
        s = self._current_session()
        if s is None:
            return
        region = s.add_region(x, y, w, h)
        self._roi_items[region.idx] = self.canvas.add_roi(region)
        self._add_card(s, region)
        self.scroll.ensureVisible(0, self.cards_container.height())
        self._update_status()

    def on_remove_region(self, sid: int, idx: int):
        session = self._session_by_sid(sid)
        if session is None:
            return
        removed = session.remove_region(idx)
        if removed is None:
            return
        # 仅当被删除的区域属于当前显示的图片时，才同步画布上的 ROI
        if self._current_session() is session:
            item = self._roi_items.pop(idx, None)
            if item is not None:
                self.canvas.remove_roi(item)
        self._remove_card(sid, idx)
        self._update_status()

    def clear_current(self):
        s = self._current_session()
        if s is None:
            return
        for r in list(s.regions):
            self._remove_card(s.sid, r.idx)
        s.regions.clear()
        s.next_idx = 1
        self._rebuild_rois()
        self._update_status()

    def clear_all_results(self):
        if not any(s.regions for s in self.sessions):
            return
        for s in self.sessions:
            for r in list(s.regions):
                self._remove_card(s.sid, r.idx)
            s.regions.clear()
            s.next_idx = 1
        self._rebuild_rois()
        self._update_status()

    # ---- 切换 ----
    def prev_image(self):
        if self.current > 0:
            self.current -= 1
            self.show_current()

    def next_image(self):
        if self.current < len(self.sessions) - 1:
            self.current += 1
            self.show_current()

    def _update_image_title(self):
        s = self._current_session()
        if s is None:
            self.image_title.setText("请打开图片")
        else:
            self.image_title.setText(
                f"  {os.path.basename(s.path)}    （拖拽框选区域，松开即出频谱；滚轮缩放）"
            )

    def _update_actions(self):
        has = self._current_session() is not None
        n = len(self.sessions)
        self.act_prev.setEnabled(has and self.current > 0)
        self.act_next.setEnabled(has and self.current < n - 1)
        self.act_clear.setEnabled(has and bool(s.regions) if (s := self._current_session()) else False)
        self.act_clear_all.setEnabled(any(s.regions for s in self.sessions))
        self.act_export.setEnabled(any(s.regions for s in self.sessions))
        self.panel.setEnabled(has)

    def _update_status(self):
        s = self._current_session()
        n = len(self.sessions)
        total = sum(len(x.regions) for x in self.sessions)
        if s is None:
            self.statusBar().showMessage("未加载图片")
        else:
            msg = f"图片 {self.current + 1}/{n}    当前 {len(s.regions)} 个区域    全部结果 {total} 个"
            if not s.adjust.is_default():
                msg += "    [已增强]"
            self.statusBar().showMessage(msg)
        self._update_actions()

    def export_results(self):
        if not any(s.regions for s in self.sessions):
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not out_dir:
            return
        csv_path = os.path.join(out_dir, "regions.csv")
        total = 0
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["image_stem", "image_path", "region_idx", "x", "y", "w", "h",
                 "patch_w", "patch_h", "spectrum_w", "spectrum_h",
                 "brightness", "contrast", "gamma", "sharpen", "clahe", "invert"]
            )
            for s in self.sessions:
                if not s.regions:
                    continue
                sub = os.path.join(out_dir, s.stem)
                os.makedirs(sub, exist_ok=True)
                adj = s.adjust
                for r in s.regions:
                    patch_path = os.path.join(sub, f"r{r.idx}_patch.png")
                    spec_path = os.path.join(sub, f"r{r.idx}_spectrum.png")
                    cv2.imwrite(patch_path, r.patch_gray)
                    cv2.imwrite(spec_path, cv2.cvtColor(r.spectrum_rgb, cv2.COLOR_RGB2BGR))
                    writer.writerow([
                        s.stem, s.path, r.idx, r.x, r.y, r.w, r.h,
                        r.patch_gray.shape[1], r.patch_gray.shape[0],
                        r.spectrum_rgb.shape[1], r.spectrum_rgb.shape[0],
                        adj.brightness, adj.contrast, round(adj.gamma, 3),
                        adj.sharpen, round(adj.clahe, 2), int(adj.invert),
                    ])
                    total += 1
        QMessageBox.information(
            self, "导出完成",
            f"已导出 {total} 个区域\n频谱图保存至各图片子目录\n坐标+增强参数汇总：{csv_path}",
        )

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left and self.current > 0:
            self.prev_image()
        elif e.key() == Qt.Key_Right and self.current < len(self.sessions) - 1:
            self.next_image()
        else:
            super().keyPressEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
