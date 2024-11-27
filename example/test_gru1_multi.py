from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid_multi import xvalid_fit_multi

class GRUBuilder1(ModelBuilder):

    def __init__(self,input_names,output_names,ndays):
        super().__init__(input_names,output_names)
        self.ntime = ndays
        self.ndays = ndays
        self.nwindows = 0
        self.window_length = 0
        self.reverse_time_inputs = False

    def build_model(self,input_layers, input_data):

        prepro_layers = self.prepro_layers(input_layers,input_data)          
        x = layers.Lambda(lambda x: tf.stack(x,axis=-1))(prepro_layers) 

        x = layers.GRU(units=32, return_sequences=True, 
                            activation='sigmoid',name='gru_1')(x)
        x = layers.GRU(units=16, return_sequences=False, 
                            activation='sigmoid',name='gru_2')(x)
        x = layers.Flatten()(x)
        
        outdim = len(self.output_names)
        # The regularization is unknown
        outputs = layers.Dense(units=outdim, name = "ec", activation='relu',
                               kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(x)
        ann = Model(inputs = input_layers, outputs = outputs)
        print(ann.summary())
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.008), 
            loss='mae',   # could be mean_absolute_error or mean_squared_error 
            metrics=['mean_absolute_error','mse'],
            run_eagerly=True
        )
        return ann


    def fit_model(self,ann,fit_input,fit_output,test_input,test_output,nepochs=80):  # ,tcb):
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=nepochs,
            batch_size=32,
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
            )
        return history, ann





def test_xvalid_mlp():
    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy","sf_tidal_filter","dcc","smscg"]
    output_names = ["x2","mal","nsl2","bdl","cse","emm2","tms","jer","sal","bac","oh4"]
    plot_locs = ["x2","cse","emm2","jer","bdl","sal","bac"]
    builder = GRUBuilder1(input_names=input_names,output_names=output_names,ndays=80)


    fpattern = "schism_base_*.csv"
    df = read_data(fpattern)
    df_in, df_out = builder.xvalid_time_folds(df,target_fold_len='180d',split_in_out=True)   # adds a column called 'fold'


    # Heuristic scaling of outputs based on known orders of magnitude
    for col in output_names:
        if col == "x2":
            df_out.loc[:,col] = df_out.loc[:,col]/100. 
        elif col in ["mal","cse","bdl","nsl2"]:
            df_out.loc[:,col] = df_out.loc[:,col]/10000.
        elif col in ["emm2","jer","tms"]:
            df_out.loc[:,col] = df_out.loc[:,col]/2000.
        else:
            df_out.loc[:,col] = df_out.loc[:,col]/1000.


    #xvalid_fit(df_in,df_out,builder,plot_folds=[0,1],plot_locs=plot_locs)
    xvalid_fit_multi(df_in,df_out,builder,plot_folds="all",plot_locs=plot_locs,out_prefix="output/schism_mlp1m.mae",nepochs=150,pool_size=12)



if __name__ == "__main__":
    test_xvalid_mlp()