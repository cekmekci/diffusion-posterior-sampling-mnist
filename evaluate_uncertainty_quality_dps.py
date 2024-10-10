import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import glob
import matplotlib.pyplot as plt
import gc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import uncertainty_toolbox as uct


def save_uncertainty_image(array, save_dir):
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(array, cmap="jet", aspect = 1.0)
    plt.axis('off')
    axins = inset_axes(ax, width = "5%", height = "100%", loc = 'lower left',
                   bbox_to_anchor = (1.02, 0., 1, 1), bbox_transform = ax.transAxes,
                   borderpad = 0)
    cbar = fig.colorbar(im, cax = axins)
    cbar.set_ticks(np.linspace(array.min(), array.max(), 6))
    cbar.set_ticklabels(['{:.4f}'.format(x) for x in np.linspace(array.min(), array.max(), 6)])
    plt.savefig(save_dir, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.cla()
    plt.clf()
    plt.close('all')
    gc.collect()

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

    # Path of the text file containing all the resulting metrics (TO-DO)
    output_file = os.path.join(output_dir, "uncertainty_results.txt")

    # Open the output file in write mode
    with open(output_file, 'w') as f:
        for model_name in model_names:
            # Collect the predictive means, predictive stds and ground truths
            pred_means = []
            pred_stds = []
            gts = []
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
                pred_mean = data['pred_mean'] # Shape (32, 32)
                pred_std = data['pred_var'] ** 0.5 # Shape (32, 32)
                # Save the corrupted image and ground truth image
                measurement_path = f"{test_example_index}_{model_name}_measurement.pdf"
                measurement_path = os.path.join(output_dir, measurement_path)
                save_image(measurement, measurement_path)
                ground_truth_path = f"{test_example_index}_{model_name}_ground_truth.pdf"
                ground_truth_path = os.path.join(output_dir, ground_truth_path)
                save_image(gt, ground_truth_path)
                # Save the predictive mean
                pred_mean_ssim_value = ssim(gt, pred_mean, data_range = 1.0)
                pred_mean_ssim_image_path = f"{test_example_index}_{model_name}_pred_mean_ssim_{pred_mean_ssim_value:.4f}.pdf"
                pred_mean_ssim_image_path = os.path.join(output_dir, pred_mean_ssim_image_path)
                save_image(pred_mean, pred_mean_ssim_image_path)
                # Append the containers
                pred_means.append(pred_mean)
                pred_stds.append(pred_std)
                gts.append(gt)
                # Save the uncertainty map
                uncertainty_map_path = f"{test_example_index}_{model_name}_pred_std.pdf"
                uncertainty_map_path = os.path.join(output_dir, uncertainty_map_path)
                save_uncertainty_image(pred_std, uncertainty_map_path)
            # Convert the lists to numpy arrays
            pred_means = np.array(pred_means)
            pred_stds = np.array(pred_stds)
            gts = np.array(gts)
            # Flatten the arrays
            pred_means_flattened = pred_means.flatten()
            pred_stds_flattened = pred_stds.flatten()
            gts_flattened = gts.flatten()
            # Calculate the uncertainty metrics
            uq_metrics = uct.metrics.get_all_metrics(
                pred_means_flattened, pred_stds_flattened, gts_flattened, verbose = True
            )
            # Save the results to a txt file
            f.write(f"Model: {model_name}\n")
            f.write(str(uq_metrics) + "\n")
            f.write("=" * 50 + "\n\n")


if __name__ == '__main__':
    # Specify the folder containing the npz files and the output file for results
    npz_folder = 'results/inpainting/'
    output_dir = 'dps_inpainting_uncertainty_performance/'
    process_npz_files(npz_folder, output_dir)
