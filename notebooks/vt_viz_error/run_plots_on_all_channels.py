import logging
from sdo.viz.compare_models import load_pred_and_gt, create_df_combined_plots, create_combined_plots

def main(): 
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    data_inventory = '/home/Valentina/inventory_1904.pkl'
    results_path = '/fdl_sdo_data/bucket2/EXPERIMENT_RESULTS/VIRTUAL_TELESCOPE'
    output_path = '/home/Valentina/results/plots/'
    # keep this low for testing
    frac_pixels = 1.0
    
    # the first element of the list is the run with root scaling, 
    # the second is the normal one
    dict_exp = {'211': 
                [ '/vale_exp_23/0600_vale_exp_23_test_predictions.npy',
                 '/vale_exp_20/0600_vale_exp_20_test_predictions.npy',
                 (-1.2, 4.5)],
                 #(-4, 2)],
                 '193': 
                 [ '/vale_exp_25/0600_vale_exp_25_test_predictions.npy',
                   '/vale_exp_13bis/0600_vale_exp_13bis_test_predictions.npy',
                  (-1.2, 4.5)],
                 #(-5, 1.2)],
                 '171': 
                 [ '/vale_exp_26/0600_vale_exp_26_test_predictions.npy',
                   '/vale_exp_14bis/0600_vale_exp_14bis_test_predictions.npy',
                  (-1.2, 4.5)],
                 #(-5, 1.2)],
                '94': 
                [ '/vale_exp_27/0400_vale_exp_27_test_predictions.npy',
                 '/vale_exp_18/0600_vale_exp_18_test_predictions.npy',
                 (-1.2, 4.5)],
                 #(-3, 3.5)],
               }
    scaling_factors = {'94': 10, '171': 2000, '193': 3000, '211': 1000, }
    for key in dict_exp.keys():
        logging.info(f'------- Channel {key} -------')
        pred1_path = results_path +  dict_exp[key][0]
        pred2_path = results_path + dict_exp[key][1]
        logging.info('Loading predictions')
        Y1_test, Y1_pred = load_pred_and_gt(pred1_path, revert_root=True, frac=frac_pixels)
        Y2_test, Y2_pred = load_pred_and_gt(pred2_path, revert_root=False, frac=frac_pixels)
        import pdb; pdb.set_trace()
        logging.info('Reverting constant scaling')
        alpha = scaling_factors[key]
        Y1_test = Y1_test * alpha
        Y1_pred = Y1_pred * alpha
        Y2_test = Y2_test * alpha
        Y2_pred = Y2_pred * alpha
        logging.info('create df1 combined_plots')
        df1, df1_q = create_df_combined_plots(Y1_test, Y1_pred, xrange=dict_exp[key][2], 
                                              val_col = 'log(YTest)-log(YPred)')
        logging.info('create df2 combined_plots')
        df2, df2_q = create_df_combined_plots(Y2_test, Y2_pred, xrange=dict_exp[key][2],
                                              val_col='log(YTest)-log(YPred)')
        logging.info('Plotting')
        create_combined_plots(key, output_path, df1, df1_q, df2_q, xrange=dict_exp[key][2],
                              val_col= 'log(YTest)-log(YPred)',
                              label1 = 'root', label2 = 'no_root')
    
if __name__ == "__main__":
    main()