import logging
from sdo.viz.compare_models import load_images_pred_and_gt_npz
import numpy as np
from sdo.metrics.correlation import pixel2pixelcor

def main(): 
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    results_path = '/fdl_sdo_data/bucket/EXPERIMENT_RESULTS/VIRTUAL_TELESCOPE'
    output_path = '/home/luiz0992/results/plots/'
    
    # the first element of the list is the run with root scaling, 
    # the second is the normal one
    dict_exp = {'211': 
                [ '/vale_exp_23/0600_vale_exp_23_test_predictions_timestamps.npz',
                 '/vale_exp_20/0600_vale_exp_20_test_predictions_timestamps.npz',
                 (-4, 2)],
                 '193': 
                 [ '/vale_exp_25/0600_vale_exp_25_test_predictions_timestamps.npz',
                   '/vale_exp_13bis/0600_vale_exp_13bis_test_predictions_timestamps.npz',
                 (-5, 1.2)],
                 '171': 
                 [ '/vale_exp_26/0600_vale_exp_26_test_predictions_timestamps.npz',
                   '/vale_exp_14bis/0600_vale_exp_14bis_test_predictions_timestamps.npz',
                 (-5, 1.2)],
                '094': 
                [ '/vale_exp_27/0400_vale_exp_27_test_predictions_timestamps.npz',
                 '/vale_exp_18/0600_vale_exp_18_test_predictions_timestamps.npz',
                 (-3, 3.5)],
               }
    for key in dict_exp.keys():
        logging.info(f'------- Channel {key} -------')
        pred1_path = results_path +  dict_exp[key][0]
        pred2_path = results_path + dict_exp[key][1]
        logging.info('Loading predictions')
        Y1_test, Y1_pred = load_images_pred_and_gt_npz(pred1_path, revert_root=True)
        Y2_test, Y2_pred = load_images_pred_and_gt_npz(pred2_path, revert_root=False)
        
        corr = np.empty([len(Y2_test),2])
        average_corr = np.empty([2])
        std_corr = np.empty([2])
        
        print(Y1_test.shape)
        
        for img_num in range(len(Y2_test)):
          corr[img_num,0] = pixel2pixelcor(Y1_test[img_num,:,:], Y1_pred[img_num,:,:])
          corr[img_num,1] = pixel2pixelcor(Y2_test[img_num,:,:], Y2_pred[img_num,:,:])
          
        # import pdb; pdb.set_trace()
        average_corr[0] = np.nanmean(corr[:,0])
        average_corr[1] = np.nanmean(corr[:,1])
        
        std_corr[0] = np.nanstd(corr[:,0])
        std_corr[1] = np.nanstd(corr[:,1])

        print(average_corr)
        print(std_corr)
    
if __name__ == "__main__":
    main()