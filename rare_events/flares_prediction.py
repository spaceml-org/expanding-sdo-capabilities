"""
This script is designed to compute some variables before and after a series of flares and save:
* the predicted images at different timestamps
* plots of the reconstruction error, on the image and on the covariance, vs timestamps
* video of the real image + reconstruction error on covariance

In its current format the script assumes the reconstructed channel is 211.
"""

import os
import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
import matplotlib.backends.backend_pdf
import moviepy.video.io.ImageSequenceClip
import img2pdf
from PIL import Image

from sdo.datasets.virtual_telescope_sdo_dataset import VirtualTelescopeSDO_Dataset
from sdo.models.vt_models.vt_unet import VT_UnetGenerator
from sdo.metrics.covariance import cov_1d, neighbor_cov
from sdo.datasets.dates_selection import get_datetime

import sunpy.visualization.colormaps as cm

logger = logging.getLogger(__name__)
logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")

def main():
    # choose parameters
    flares_path = '/home/Valentina/expanding-sdo-capabilities/rare_events/flares_modelling.csv'
    results_path = '/fdl_sdo_data/bucket/EXPERIMENT_RESULTS/VIRTUAL_TELESCOPE/vale_exp_20/'
    model_path = results_path + '0600_vale_exp_20_model.pth'
    output_folder = '/fdl_sdo_data/bucket/EXPERIMENT_RESULTS/VIRTUAL_TELESCOPE/vale_exp_20/flares_predictions/'
    n_flares = 20
    orig_channel = '211'
    
    # select timestamps for the experiment (2h30 of buffer added before and after each  start/end time)
    flares_df = pd.read_csv(flares_path)
    df_selection = flares_df[:n_flares][['start_time', 'end_time']]
    df_selection['start_datetime'] = df_selection['start_time'].apply(lambda x: get_datetime(x, 2, 30, add=False))
    df_selection['end_datetime'] = df_selection['end_time'].apply(lambda x: get_datetime(x, 2, 30, add=True))

    # load the model
    logger.info ('Load the model')
    model = VT_UnetGenerator(input_shape=[3, 512, 512])
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    run_flares = []
    for idx_flare in range(n_flares):
        logger.info(f'Event {idx_flare + 1}/{n_flares}')
        output_path = output_folder + f'flare_{idx_flare}/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        try:
            # load the data
            datetime_range = [[str(df_selection.loc[idx_flare].start_datetime), str(df_selection.loc[idx_flare].end_datetime)]]
            data = VirtualTelescopeSDO_Dataset(data_basedir='/fdl_sdo_data/SDOMLmm/fdl_sw/SDOMLmm',
                                               data_inventory='/home/Valentina/inventory_1904.pkl',
                                               instr=["AIA", "AIA", "AIA", "AIA"],
                                               num_channels = 4,
                                               channels=["0094", "0193", "0171", "0211"],
                                               datetime_range = datetime_range,
                                               resolution=512,
                                               subsample=1,
                                               test=False,
                                               test_ratio=0.0,
                                               shuffle=False,
                                               normalization=0,
                                               scaling=True,
                                               apodize=False,
                                               holdout=False)
            # compute predictions (or load if available)
            input_data = []
            gt_img = []
            timestamps = []
            outputs = []
            for idx, _ in enumerate(data.timestamps):
                input_data.append(data[idx][0])
                gt_img.append(data[idx][1])
                timestamps.append(data[idx][2])
            filename = output_path  + f'flares_{idx_flare}_outputs.rt'
            if os.path.isfile(filename):
                logger.info('Loading predictions')
                outputs = torch.load(filename)
                if len(outputs) != len(data.timestamps):
                    logger.error("Error, length of the saved files doesn't match")
            else:
                logger.info('Computing predictions')
                for idx, _ in enumerate(data.timestamps):
                    logger.info(f'Computing timestamp: {data[idx][2]}')
                    outputs.append(
                      model(data[idx][0].unsqueeze(0)).detach().numpy().reshape(1, 512, 512)
                    )
                # save predictions if they were not already available
                filename = output_path  + f'flares_{idx_flare}_outputs.rt'
                torch.save(outputs, filename)

            # compute and save reconstruction errors
            logger.info('Computing reconstruction errors')
            results = pd.DataFrame()
            results_cov = pd.DataFrame()
            for i, _ in enumerate(data.timestamps):
                ts = '-'.join([str(i.item()) for i in timestamps[i]])
                X_s = outputs[i].reshape(512, 512)
                X_orig = gt_img[i].detach().numpy().reshape(512, 512)
                flux_orig = X_orig.sum()
                diff_orig = (X_s - X_orig).sum()
                percent_diff_orig = (X_s - X_orig).sum() * 100 / X_orig.sum()
                # reconstruction error on image
                results = results.append(pd.DataFrame(index=[i], data={'Timestamp': ts, 'Channel': orig_channel,
                                                                       'Flux': flux_orig,
                                                                       'Diff': diff_orig,
                                                                       '%Diff': percent_diff_orig}
                                                    )
                                       )
                # reconstruction error on co-variance
                for j, ch in enumerate(["094", "193", "171"]):
                    cov_orig, cov_synth = 0, 0
                    Y = input_data[i][j].detach().numpy().reshape(512, 512)
                    cov_orig = cov_1d(X_orig, Y)
                    cov_synth = cov_1d(X_s, Y)
                    flux = input_data[i][j].detach().numpy().sum()
                    results_cov = results_cov.append(
                        pd.DataFrame(index=[i], data={'Timestamp': ts, 'Channel': ch,
                                                    'Flux': flux,
                                                    'True Cov': cov_orig, 'Pred Cov': cov_synth,
                                                    'Diff': (cov_orig - cov_synth),
                                                    '%Diff': (cov_orig - cov_synth) * 100 / cov_orig})
                    )
            results_cov.reset_index().drop('index', axis=1).to_csv(output_path + 'results.csv')
            results_cov.reset_index().drop('index', axis=1).to_csv(output_path + 'results_cov.csv')

            # plot reconstruction errors and total flux
            logger.info('Plotting reconstruction errors')
            fig, axs = plt.subplots(3, 1)
            for channel in ["094", "193", "171"]:
                tmp_results = results_cov[results_cov.Channel == channel]
                factor = tmp_results.Flux.min()
                tmp_results.Flux = tmp_results.Flux / factor
                tmp_results.plot(x='Timestamp', y='Flux', label=channel, figsize=(17, 10), ax=axs[0])
            factor = results.Flux.min()
            results.Flux = results.Flux / factor
            results.plot(x='Timestamp', y='Flux', label='211', figsize=(17, 10), ax=axs[0])
            axs[0].set_ylabel('Total Flux')
            axs[0].set_title('Flux by Channel /normalized to the min')

            results.plot(x='Timestamp', y='%Diff', label='211', figsize=(17, 10), ax=axs[1])
            axs[1].set_ylabel('Real-Synt Image')
            axs[1].set_title('Reconstruction Error on Image')

            for channel in ["094", "193", "171"]:
                results_cov[results_cov.Channel == channel].plot(x='Timestamp', y='%Diff', label=channel,
                                                               figsize=(17, 10), ax=axs[2])
            axs[2].set_ylabel('Real-Synt Cov')
            axs[2].set_title('Reconstruction Error on Covariance')
            fig.tight_layout()
            filename = output_path + 'Errors_vs_timestamps.png'
            plt.savefig(filename, bbox_inches='tight')

            # compute covariance map with rolling window
            logger.info('Computing covariance maps')
            cov_window = 15
            filename = output_path + 'Covariance_maps.pdf'
            pdf = matplotlib.backends.backend_pdf.PdfPages(filename)
            for i, _ in enumerate(data.timestamps):
                # taking one timestamp every 2 to reduce the size of the plot
                if i%2 == 0:
                    fig, axs = plt.subplots(1, 5, figsize=(15, 10))
                    str_time = '_'.join([str(number) for number in data.timestamps[i]])
                    X_s = outputs[i].reshape(512, 512)
                    X_orig = gt_img[i].detach().numpy().reshape(512, 512)
                    im = axs[0].set_title(f'AIA 211 GT')
                    im = axs[0].imshow(X_orig, origin='lower', cmap=plt.get_cmap('sdoaia211'))
                    im = axs[1].set_title(f'{str_time} AIA 211 PR - GT')
                    im = axs[1].imshow((X_s - X_orig), cmap='seismic', origin='lower', vmin=-0.8, vmax=0.8)
                    for j, ch in enumerate(["094", "193", "171"]):
                        Y = input_data[i][j].detach().numpy().reshape(512, 512)
                        cov_synth = neighbor_cov(X_s, Y, size=cov_window)
                        cov_orig = neighbor_cov(X_orig, Y, size=cov_window)
                        im = axs[j + 2].set_title(f'Cov AIA {ch}-{211} PR - GT')
                        im = axs[j + 2].imshow((cov_synth - cov_orig), cmap='seismic', origin='lower', vmin=-0.8, vmax=0.8)
                        fig.subplots_adjust(bottom=0.1, right=0.8, top=0.8)
                        cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.3])
                        fig.colorbar(im, cax=cbar_ax)
                    #plt.show()
                    pdf.savefig(fig)
            
            #plt.savefig(filename, bbox_inches='tight')
            #plt.close()
            pdf.close()
            run_flares.append(idx_flare)
            
            # as above but only for 211-094 covariance and save as single image to create a video
            logger.info('Creating video')
            video_folder = output_path + 'video/'
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            fig_format = '.png'
            fps = 2
            cmap = plt.get_cmap('sdoaia094')
            for i, timestamp in enumerate(data.timestamps):
                fig, axs = plt.subplots(1,2, figsize=(15, 10))
                filename = video_folder + '_'.join([str(number) for number in timestamp]) + fig_format
                X_s = outputs[i].reshape(512, 512)
                X_orig = gt_img[i].detach().numpy().reshape(512, 512)
                input_94 = input_data[i][0].reshape(512, 512)
                axs[0].set_title(f'{timestamp} AIA 094 GT')
                im = axs[0].imshow(input_94, origin='lower', cmap=cmap)
                for j, ch in enumerate(["094"]):
                    Y = input_data[i][j].detach().numpy().reshape(512,512)
                    cov_synth = neighbor_cov(X_s, Y, size=10)
                    cov_orig = neighbor_cov(X_orig, Y, size=10)
                    axs[j+1].set_title(f'Cov AIA {ch}-{211} PR - GT')
                    im = axs[j+1].imshow((cov_synth - cov_orig), cmap='seismic', origin='lower', vmin=-0.8, vmax=0.8)
                    fig.subplots_adjust(bottom=0.1, right=0.8, top=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.3])
                    fig.colorbar(im, cax=cbar_ax)
                plt.savefig(filename, bbox_inches='tight')
                plt.close()

            video_name = output_path + f'flare_{idx_flare}.mp4'
            image_files = [video_folder + img for img in os.listdir(video_folder) if img.endswith(".png")]
            image_files.sort()
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
            clip.write_videofile(video_name)
            
        except Exception as e:
            logger.info(f'Flare {idx_flare} did not complete. See error {e}')

    # aggregate plots
    logger.info(f'Aggregated files will include {len(run_flares)} flares.')
    l_converted_img = []
    for idx_flare in run_flares:
        output_path = output_folder + f'flare_{idx_flare}/'
        png_filename = output_path + 'Errors_vs_timestamps.png'
        pdf_filename = output_path + 'Errors_vs_timestamps.pdf'
        rgba = Image.open(png_filename)
        rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # white background
        rgb.paste(rgba, mask=rgba.split()[3])               # paste using alpha channel as mask
        rgb.save(pdf_filename, 'PDF', resoultion=100.0)
        l_converted_img.append(rgb)
       
    l_converted_img[0].save(output_folder + "summary_1dplots.pdf",save_all=True, 
             append_images=l_converted_img[1:])
    logger.info(f'summary_1dplots.pdf created.')
        
    
if __name__ == "__main__":
    main()
    
