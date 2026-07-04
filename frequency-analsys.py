import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_local_frequency(image_path):
    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return

    print("=== 频域局部对比分析工具 ===")
    print("1. 请在弹出的图片窗口中，用鼠标拖拽框选【目标区域】。")
    print("2. 框选完成后，按 [SPACE] 或 [ENTER] 键确认。")
    
    # 交互式框选目标区域
    roi_target = cv2.selectROI("Select Target Area (Press SPACE/ENTER to confirm)", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    print("3. 现在，请在弹出的图片窗口中，用鼠标拖拽框选【容易混淆的背景区域】。")
    print("4. 框选完成后，按 [SPACE] 或 [ENTER] 键确认。")
    
    # 交互式框选背景区域
    roi_bg = cv2.selectROI("Select Background Area (Press SPACE/ENTER to confirm)", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # 解析 ROI 坐标并裁剪 (x, y, width, height)
    x_t, y_t, w_t, h_t = roi_target
    x_b, y_b, w_b, h_b = roi_bg

    # 裁剪并转换为灰度图
    target_patch = cv2.cvtColor(img[y_t:y_t+h_t, x_t:x_t+w_t], cv2.COLOR_BGR2GRAY)
    bg_patch = cv2.cvtColor(img[y_b:y_b+h_b, x_b:x_b+w_b], cv2.COLOR_BGR2GRAY)

    # 定义 FFT 计算函数
    def compute_spectrum(patch):
        # 如果裁剪的区域太小，FFT 效果不佳，这里可以做个简单的放大（可选）
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_LINEAR)
            
        f = np.fft.fft2(patch)
        fshift = np.fft.fftshift(f)
        # 计算幅度谱并取对数以便可视化
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        return magnitude_spectrum

    target_spec = compute_spectrum(target_patch)
    bg_spec = compute_spectrum(bg_patch)

    # 可视化对比结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行：目标分析
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].add_patch(plt.Rectangle((x_t, y_t), w_t, h_t, edgecolor='lime', facecolor='none', lw=3))
    axes[0, 0].add_patch(plt.Rectangle((x_b, y_b), w_b, h_b, edgecolor='red', facecolor='none', lw=3))
    axes[0, 0].set_title("Original Image (Green: Target, Red: Background)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_patch, cmap='gray')
    axes[0, 1].set_title("Target Patch (Spatial)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(target_spec, cmap='inferno') # 使用暖色调凸显高频
    axes[0, 2].set_title("Target Spectrum (Frequency)")
    axes[0, 2].axis('off')

    # 第二行：背景分析
    axes[1, 0].axis('off') # 留空
    
    axes[1, 1].imshow(bg_patch, cmap='gray')
    axes[1, 1].set_title("Background Patch (Spatial)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(bg_spec, cmap='inferno')
    axes[1, 2].set_title("Background Spectrum (Frequency)")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 示例图像路径，请根据实际情况修改
    image_path = r"E:\refac_newbase\samples_20_per_label\20260512102447.jpg"
    analyze_local_frequency(image_path)
