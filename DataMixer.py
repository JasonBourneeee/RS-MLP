import os
import numpy as np
import torch
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import random
import itertools

class SpectralDataset:
    def __init__(self, spectra, concentrations):
        self.spectra = torch.FloatTensor(spectra)
        self.concentrations = torch.FloatTensor(concentrations)
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return self.spectra[idx], self.concentrations[idx]

class HybridSpectralMixer:
    def __init__(self, base_path="./DataMixer/Prue"):
        self.base_path = base_path
        self.compounds = ['DIMP', 'DMMP', 'TEP']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wavenumbers = None
        
        # 定义二元和三元混合浓度组合
        self.binary_intervals = [[75, 25], [50, 50], [25, 75], [10, 90], [5, 95], [1, 99], [0.5, 99.5]]
        
        # 定义三元混合的基础比例组合，包含极端情况
        self.ternary_ratios = [
            # 常规组合
            [60, 30, 10],
            [60, 10, 30],
            [30, 60, 10],
            [30, 10, 60],
            [10, 60, 30],
            [10, 30, 60],
            
            # 添加极端比例组合
            [90, 9, 1],
            [90, 1, 9],
            [9, 90, 1],
            [9, 1, 90],
            [1, 90, 9],
            [1, 9, 90],
            
            # 中等-极端组合
            [80, 15, 5],
            [80, 5, 15],
            [15, 80, 5],
            [15, 5, 80],
            [5, 80, 15],
            [5, 15, 80],
            
            # 近似均匀组合
            [40, 35, 25],
            [40, 25, 35],
            [35, 40, 25],
            [35, 25, 40],
            [25, 40, 35],
            [25, 35, 40]
        ]
        
        self.compound_colors = {'DIMP': 'red', 'DMMP': 'blue', 'TEP': 'green'}
        print(f"Using device: {self.device}")

    def denoise_spectrum(self, spectrum):
        """改进的去噪处理，保持峰的锐度"""
        # 使用较小的中值滤波核，以保持峰的形状
        spectrum_medfilt = medfilt(spectrum, kernel_size=3)
        
        # 使用较小的窗口长度，以避免过度平滑
        window_length = min(7, len(spectrum_medfilt) - 1)
        if window_length % 2 == 0:
            window_length -= 1
            
        # 使用较低阶数的多项式拟合，以保持峰的特征
        spectrum_smooth = savgol_filter(spectrum_medfilt, 
                                    window_length=window_length, 
                                    polyorder=2)
        
        return gaussian_filter1d(spectrum_smooth, sigma=0.3)

    def load_all_data(self):
        """加载纯物质光谱数据并取平均值"""
        pure_spectra = {compound: [] for compound in self.compounds}
        
        for compound in self.compounds:
            compound_path = os.path.join(self.base_path, compound + '100%')
            for file in os.listdir(compound_path):
                if file.startswith('1000ms_') and file.endswith('.txt'):
                    filepath = os.path.join(compound_path, file)
                    wavenumbers, intensity = np.loadtxt(filepath, delimiter='\t', unpack=True)
                    if self.wavenumbers is None:
                        self.wavenumbers = wavenumbers
                    spectrum = self.denoise_spectrum(intensity)
                    pure_spectra[compound].append(spectrum)
        
        averaged_spectra = {compound: np.mean(pure_spectra[compound], axis=0) for compound in self.compounds}
        return averaged_spectra 

    def generate_binary_mixtures(self, averaged_spectra):
        for base_idx, base_compound in enumerate(self.compounds):
            for second_idx in range(base_idx + 1, len(self.compounds)):
                second_compound = self.compounds[second_idx]
                
                for interval in self.binary_intervals:
                    concentrations = [0] * 3
                    concentrations[base_idx] = interval[0] / 100
                    concentrations[second_idx] = interval[1] / 100

                    mixed_spectrum = np.zeros(len(averaged_spectra[self.compounds[0]]))
                    for idx, conc in enumerate(concentrations):
                        if conc > 0:
                            # 使用非线性响应函数，但保持峰的独立性
                            # 降低非线性强度以避免峰展宽
                            response = conc * (1 - 0.05 * conc)  # 减小非线性影响
                            spectrum = averaged_spectra[self.compounds[idx]]
                            mixed_spectrum += spectrum * response

                    # 使用更小的高斯平滑参数，以保持峰的锐度
                    mixed_spectrum = gaussian_filter1d(mixed_spectrum, sigma=0.3)
                    yield mixed_spectrum, concentrations

    def generate_ternary_mixtures(self, averaged_spectra):
        """生成三元混合光谱，保持各个峰的独立性"""
        for ratio in self.ternary_ratios:
            # 转换浓度比例为小数
            concentrations = [ratio[i] / 100 for i in range(3)]
            
            mixed_spectrum = np.zeros(len(averaged_spectra[self.compounds[0]]))
            
            # 第一步：添加各组分的贡献，保持峰的独立性
            for idx, conc in enumerate(concentrations):
                if conc > 0:
                    # 使用更温和的非线性响应函数，避免峰展宽
                    response = conc * (1 - 0.05 * conc)  # 减小非线性强度
                    mixed_spectrum += averaged_spectra[self.compounds[idx]] * response

            active_compounds = [(idx, conc) for idx, conc in enumerate(concentrations) if conc > 0]
            if len(active_compounds) > 1:
                for i in range(len(active_compounds)):
                    for j in range(i + 1, len(active_compounds)):
                        idx1, conc1 = active_compounds[i]
                        idx2, conc2 = active_compounds[j]
                        
                        # 减小交互效应的强度，避免峰展宽
                        interaction = (conc1 * conc2 * 0.02 *  # 进一步降低交互系数
                                    (averaged_spectra[self.compounds[idx1]] * 
                                    averaged_spectra[self.compounds[idx2]]) / 
                                    (np.max(averaged_spectra[self.compounds[idx1]]) * 4))  # 增加归一化因子
                        
                        # 只在各自峰的位置添加微弱交互效应
                        peaks1 = averaged_spectra[self.compounds[idx1]] > 0.1 * np.max(averaged_spectra[self.compounds[idx1]])
                        peaks2 = averaged_spectra[self.compounds[idx2]] > 0.1 * np.max(averaged_spectra[self.compounds[idx2]])
                        mask = peaks1 | peaks2
                        mixed_spectrum[mask] += interaction[mask] * 0.5  # 减半交互效应强度

            # 使用更小的高斯平滑参数，以保持峰的锐度
            mixed_spectrum = gaussian_filter1d(mixed_spectrum, sigma=0.3)
            yield mixed_spectrum, concentrations


    def add_noise(self, spectrum, noise_level=0.01):
        noise = np.random.normal(0, noise_level, size=spectrum.shape)
        return np.maximum(spectrum + noise, 0)

    def plot_mixture_comparison(self, folder_path, output_path, averaged_spectra):
        spectrum_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not spectrum_files:
            return
        
        random_file = random.choice(spectrum_files)
        file_path = os.path.join(folder_path, random_file)
        
        _, mixed_spectrum = np.loadtxt(file_path, delimiter='\t', unpack=True)
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(self.wavenumbers, mixed_spectrum, 'k-', label='Mixed Spectrum', linewidth=2)
        
        for compound in self.compounds:
            plt.plot(self.wavenumbers, averaged_spectra[compound], 
                     color=self.compound_colors[compound], 
                     alpha=0.3, 
                     label=f'Pure {compound}')
        
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title(f'Spectral Comparison - {os.path.basename(folder_path)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_path, f"{os.path.basename(folder_path)}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def create_overview_plots(self, base_dir, output_path, averaged_spectra):
        plt.figure(figsize=(15, 8))
        
        binary_folders = [f for f in os.listdir(os.path.join(base_dir, "binary"))]
        for folder in binary_folders:
            folder_path = os.path.join(base_dir, "binary", folder)
            spectrum_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            if spectrum_files:
                random_file = random.choice(spectrum_files)
                file_path = os.path.join(folder_path, random_file)
                _, spectrum = np.loadtxt(file_path, delimiter='\t', unpack=True)
                plt.plot(self.wavenumbers, spectrum, label=folder, alpha=0.7)
        
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title('Overview of Binary Mixtures')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "binary_overview.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(15, 8))
        
        ternary_folders = [f for f in os.listdir(os.path.join(base_dir, "ternary"))]
        for folder in ternary_folders:
            folder_path = os.path.join(base_dir, "ternary", folder)
            spectrum_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
            if spectrum_files:
                random_file = random.choice(spectrum_files)
                file_path = os.path.join(folder_path, random_file)
                _, spectrum = np.loadtxt(file_path, delimiter='\t', unpack=True)
                plt.plot(self.wavenumbers, spectrum, label=folder, alpha=0.7)
        
        plt.xlabel('Wavenumber (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title('Overview of Ternary Mixtures')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "ternary_overview.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def save_mixture_data(self, averaged_spectra):
        plots_dir = "comparison_plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        binary_count = 0
        for mixed_spectrum, concentrations in self.generate_binary_mixtures(averaged_spectra):
            compounds_conc = [(comp, conc) for comp, conc in zip(self.compounds, concentrations) if conc > 0]
            ratio_str = '_'.join(f"{comp}_{round(conc * 100, 1)}%" for comp, conc in compounds_conc)
            
            folder_path = os.path.join("hybrid_mixed_spectra", "binary", ratio_str)
            os.makedirs(folder_path, exist_ok=True)
            
            for i in range(50):
                noisy_spectrum = self.add_noise(mixed_spectrum)
                filename = os.path.join(folder_path, f"spectrum_{i+1}.txt")
                output = np.column_stack((self.wavenumbers, noisy_spectrum))
                np.savetxt(filename, output, delimiter='\t', fmt='%.6f')
            
            self.plot_mixture_comparison(folder_path, plots_dir, averaged_spectra)
            binary_count += 1
            print(f"Saved binary mixture {binary_count}: {ratio_str}")

        ternary_count = 0
        for mixed_spectrum, concentrations in self.generate_ternary_mixtures(averaged_spectra):
            compounds_conc = [(comp, conc) for comp, conc in zip(self.compounds, concentrations) if conc > 0]
            ratio_str = '_'.join(f"{comp}_{round(conc * 100, 1)}%" for comp, conc in compounds_conc)
            folder_path = os.path.join("hybrid_mixed_spectra", "ternary", ratio_str)
            os.makedirs(folder_path, exist_ok=True)
            for i in range(50):
                noisy_spectrum = self.add_noise(mixed_spectrum)
                filename = os.path.join(folder_path, f"spectrum_{i+1}.txt")
                output = np.column_stack((self.wavenumbers, noisy_spectrum))
                np.savetxt(filename, output, delimiter='\t', fmt='%.6f')
    
            self.plot_mixture_comparison(folder_path, plots_dir, averaged_spectra)
            ternary_count += 1
        self.create_overview_plots("hybrid_mixed_spectra", plots_dir, averaged_spectra)

def main():
    mixer = HybridSpectralMixer(base_path=r"D:\GAN_CNN\Analog_mixing\Prue")
    averaged_spectra = mixer.load_all_data()
    mixer.save_mixture_data(averaged_spectra)

if __name__ == "__main__":
    main()