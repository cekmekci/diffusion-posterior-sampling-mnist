import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import glob
import matplotlib.pyplot as plt
import gc


def save_image(array, save_dir):
    plt.figure()
    plt.imshow(array, cmap="gray")
    plt.axis('off')
    plt.savefig(save_dir, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()


def process_npz_files(npz_folder, output_dir):
    # Create the output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Obtain the list of npz files
    npz_files = glob.glob(os.path.join(npz_folder, "*.npz"))
    # Collect all model names based on file naming convention
    model_names = ["ema_0.9999_040000_model1.pt", "ema_0.9999_040000_model2.pt"]
    # Path of the text file containing all the resulting metrics
    output_file = os.path.join(output_dir, "ssim_results.txt")
    # Open the output file in write mode
    with open(output_file, 'w') as f:
        for model_name in model_names:
            # Collect the SSIM between the predictive mean and ground truth
            ssim_predictive_values = []
            # Filter files corresponding to this model
            model_files = [file for file in npz_files if model_name in file]
            # Loop through each file for the current model
            for file in model_files:
                # Extract test example index and model name from the filename
                file_name = os.path.basename(file)
                test_example_index = file_name.split('_')[0] # ex: 00000
                # Load the npz file
                data = np.load(file)
                gt = data['gt']  # Shape (32, 32)
                measurement = data['measurement']  # Shape (32, 32)
                generated_samples = data['generated_samples']  # Shape (128, 32, 32)
                pred_mean = data['pred_mean'] # Shape (32, 32)
                # Save the corrupted image and ground truth image
                measurement_path = f"{test_example_index}_{model_name}_measurement.pdf"
                measurement_path = os.path.join(output_dir, measurement_path)
                ground_truth_path = f"{test_example_index}_{model_name}_ground_truth.pdf"
                ground_truth_path = os.path.join(output_dir, ground_truth_path)
                save_image(gt, ground_truth_path)
                save_image(measurement, measurement_path)
                # Calculate SSIM between the ground truth and each sample
                ssim_values = []
                for i in range(generated_samples.shape[0]):
                    sample = generated_samples[i,:,:]
                    sample_ssim = ssim(gt, sample, data_range = 1.0)
                    ssim_values.append(sample_ssim)
                ssim_values = np.array(ssim_values)
                mean_ssim = np.mean(ssim_values)
                std_ssim = np.std(ssim_values)
                # Get the min and max SSIM samples
                min_ssim_idx = np.argmin(ssim_values)
                max_ssim_idx = np.argmax(ssim_values)
                min_ssim_value = ssim_values[min_ssim_idx]
                max_ssim_value = ssim_values[max_ssim_idx]
                # Save the min and max SSIM samples
                min_ssim_image_path = f"{test_example_index}_{model_name}_min_ssim_{min_ssim_value:.4f}.pdf"
                min_ssim_image_path = os.path.join(output_dir, min_ssim_image_path)
                max_ssim_image_path = f"{test_example_index}_{model_name}_max_ssim_{max_ssim_value:.4f}.pdf"
                max_ssim_image_path = os.path.join(output_dir, max_ssim_image_path)
                save_image(generated_samples[min_ssim_idx], min_ssim_image_path)
                save_image(generated_samples[max_ssim_idx], max_ssim_image_path)
                # Calculate SSIM between the ground truth and the predictive mean
                pred_mean_ssim_value = ssim(gt, pred_mean, data_range = 1.0)
                pred_mean_ssim_image_path = f"{test_example_index}_{model_name}_pred_mean_ssim_{pred_mean_ssim_value:.4f}.pdf"
                pred_mean_ssim_image_path = os.path.join(output_dir, pred_mean_ssim_image_path)
                # Save the mean image
                save_image(pred_mean, pred_mean_ssim_image_path)
                # Save the SSIM between the predictive mean and ground truth
                ssim_predictive_values.append(pred_mean_ssim_value)
                # Write the results for the test example
                f.write(f"Model: {model_name}, Test Example: {test_example_index}\n")
                f.write(f"Min SSIM: {min_ssim_value:.4f}\n")
                f.write(f"Max SSIM: {max_ssim_value:.4f}\n")
                f.write(f"Pred Mean SSIM: {pred_mean_ssim_value:.4f}\n")
                f.write("-" * 25 + "\n")
            # Calculate average and standard deviation of the SSIM (between predictive mean and gt) for the current model
            avg_ssim = np.mean(ssim_predictive_values)
            std_ssim = np.std(ssim_predictive_values)
            # Write the summary for this model
            f.write(f"Summary for Model: {model_name}\n")
            f.write(f"Overall Average Predictive SSIM: {avg_ssim:.4f}\n")
            f.write(f"Overall Predictive SSIM Std Dev: {std_ssim:.4f}\n")
            f.write("=" * 50 + "\n\n")


if __name__ == '__main__':
    # Specify the folder containing the npz files and the output file for results
    npz_folder = 'results/inpainting/'
    output_dir = 'dps_inpainting_reconstruction_performance/'
    process_npz_files(npz_folder, output_dir)
