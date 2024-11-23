from casanntra.read_data import read_data
from casanntra.model_builder import *
from casanntra.xvalid import xvalid_fit



class GRUBuilder2(GRUModelBuilder):

    def __init__(self,input_names,output_names,ndays):
        super().__init__(input_names,output_names,ndays)
        self.reverse_time_inputs = False

    def build_model(self,input_layers, input_data):

        prepro_layers = self.prepro_layers(input_layers,input_data)          
        x = layers.Lambda(lambda x: tf.stack(x,axis=-1))(prepro_layers) 

        x = layers.GRU(units=16, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                            activation='sigmoid')(x)
        first_gru = x
        # todo: units was successful at units=12
        x = layers.GRU(units=16, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                            activation='sigmoid')(x)
        
        x = tf.keras.layers.Add()([x, first_gru])
        
        x = layers.GRU(units=16, return_sequences=False, 
                            activation='sigmoid')(x)

        x = layers.Flatten()(x)
        
        outdim = len(self.output_names)
        # The regularization is unknown
        outputs = layers.Dense(units=outdim, name = "ec", activation='relu',
                               kernel_regularizer = regularizers.l1_l2(l1=0.001,l2=0.001))(x)
        ann = Model(inputs = input_layers, outputs = outputs)
        print(ann.summary())
        ann.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=0.008), 
            loss='mse',   # todo: was successful at mae 
            metrics=['mean_absolute_error','mse'],
            run_eagerly=True
        )
        return ann


    def fit_model(self,ann,fit_input,fit_output,test_input,test_output,nepochs=200):  # ,tcb):
        history = ann.fit(
            fit_input,
            fit_output,
            epochs=nepochs,
            batch_size=32, # todo was successful at 64
            validation_data=(test_input, test_output),
            verbose=2,
            shuffle=True
            )
        return history, ann


def test_xvalid_gru():

    input_names = [ "sac_flow","exports","sjr_flow","cu_flow","sf_tidal_energy","sf_tidal_filter","dcc","smscg"]
    #output_names = ["x2","pct", "mal", "cse","anh","emm2","srv","rsl","oh4","trp","dsj","hll","bdl"]
    output_names = ["x2","mal","nsl2","bdl","cse","emm2","tms","anh","jer","sal","bac","rsl","oh4"]
    plot_locs = ["x2","cse","emm2","jer","bdl","sal","bac"]

    builder = GRUBuilder2(input_names=input_names,output_names=output_names,ndays=80)


    fpattern = "schism_base_*.csv"
    df = read_data(fpattern)

    df_in, df_out = builder.xvalid_time_folds(df,target_fold_len='180d',split_in_out=True)   # adds a column called 'fold'

    # Heuristic scaling of outputs based on known orders of magnitude
    for col in output_names:
        if col == "x2":
            df_out.loc[:,col] = df_out.loc[:,col]/100. 
        elif col in ["mal","cse","anh","bdl","nsl2"]:
            df_out.loc[:,col] = df_out.loc[:,col]/10000.
        elif col in ["emm2","jer","tms"]:
            df_out.loc[:,col] = df_out.loc[:,col]/2000.
        else:
            df_out.loc[:,col] = df_out.loc[:,col]/1000.


    xvalid_fit(df_in,df_out,builder,nepochs=50,plot_folds="all",
               plot_locs=plot_locs,out_prefix="output/schism_gru2.16.16")

if __name__ == "__main__":
   test_xvalid_gru()

