from keras.preprocessing.image import ImageDataGenerator
import numpy as np
def train_model(model, x_train, x_train_feat, y_train, x_test, x_test_feat, y_test, x_val_feat, train_batch_size, test_batch_size, epochs):
#def train_model(model, x_train, x_train_feat, y_train, x_test, x_test_feat, y_test, train_batch_size, test_batch_size, epochs, model_dir, model_name,patience=5, monitor='val_acc'):


    '''
    Training function
    '''

    train_datagen = ImageDataGenerator(
        featurewise_center=False, # also use in test gen if activated AND fit test_gen on train data
        samplewise_center=False,
        featurewise_std_normalization=False, # also use in test gen if activated AND fit test_gen on train data
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=0,
        rotation_range=0.05,
        width_shift_range=0.05,
        height_shift_range=0.05,
        channel_shift_range=0,
        fill_mode='nearest',
        cval=0,
        vertical_flip=False,
        rescale=1./255,
        shear_range=0.,
        zoom_range=0.,
        horizontal_flip=False)

    train_datagen.fit(x_train)

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_std_normalization=False,
        featurewise_center=False)

    train_generator = train_datagen.flow(
        x_train,
        y_train,
        batch_size=train_batch_size,
        shuffle=False)

    def train_feat_gen(x_train_feat, train_batch_size):
        while True:
            for batch in range(len(x_train_feat) // train_batch_size + 1):
                if batch > max(range(len(x_train_feat) // train_batch_size)):
                    #print("\n****************** 1)len is:",
                    #    x_train_feat[batch*train_batch_size:].shape,
                    #         "*****************\n")
                    yield x_train_feat[batch*train_batch_size:]
                else:
                    #print("\n****************** 2)len is:",
                    #    x_train_feat[batch*train_batch_size:(1+batch)*train_batch_size].shape,
                    #         "*****************\n")
                    yield x_train_feat[batch*train_batch_size:(1+batch)*train_batch_size]
                    #shape:(64, 2)


    def val_feat_gen(x_val_feat, test_batch_size):
        while True:

            #x_val_feat=1000
            for batch in range(len(x_val_feat) // test_batch_size + 1):
                if batch > max(range(len(x_val_feat) // test_batch_size)):
                    yield x_val_feat[batch*test_batch_size:]
                else:
                    yield x_val_feat[batch*test_batch_size:(1+batch)*test_batch_size]

    def merge_generator(gen1, gen2):
        #gen1:train_generator h' validation_generator
        #gen2:train_feat_gen h' val_feat_gen
        while True:
            #print("i is:",i)
            X1 = gen1.__next__()
            X2 = gen2.__next__()

            #print("img:",X1[0].shape,",pose:",X2.shape,"gaze:",X1[1].shape)
            
            #img: (64, 36, 60, 1) ,pose: (40, 2) gaze: (64, 2)
            if (X1[0].shape)[0] != (X1[1].shape)[0] or (X1[0].shape)[0] != (X2.shape)[0]:
                MIN= min((X1[0].shape)[0],(X1[1].shape)[0],(X2.shape)[0])
                print("\n******* ",MIN, " *******\n")
                yield ({'img_input': X1[0][:MIN], 'pose_input': X2[:MIN]}, {'gaze_output': X1[1][:MIN]})
                print("\n******* problem *******\n")
                continue

            
            yield ({'img_input': X1[0], 'pose_input': X2}, {'gaze_output': X1[1]})
            #yield [X1[0], X2], X1[1]


    validation_generator = test_datagen.flow(
        x_test,
        y_test,
        batch_size=test_batch_size
        )

    final_train_gen = merge_generator(train_generator, train_feat_gen(x_train_feat, train_batch_size))
    final_val_gen = merge_generator(validation_generator, val_feat_gen(x_val_feat, test_batch_size))

    import time
    from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard,ReduceLROnPlateau
    
    ### TODO:
    # 1)Ftiakse ena customized callback gia metriseis(gt panta miden to gaze output?)
    # link edw:
    # https://www.ir.com/blog/visualizing-outputs-cnn-model-training-phase
    # https://keras.io/callbacks/
    # https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    # https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
    # fainetai kalo
    # 2)https://www.google.com/search?client=ubuntu&channel=fs&q=keras+predict+always+0&ie=utf-8&oe=utf-8
    # 3) https://github.com/hirotong/pytorch_mpiigaze...tsekare edw ulopoihsh
    #     


    import numpy as np
    import math,keras
    from keras import backend as K
    from matplotlib import pyplot as plt
    #%matplotlib inline    
    #from IPython import get_ipython
    #get_ipython().run_line_magic('matplotlib', 'inline')
    #This function returns a list that contains the output of each kernel 
    #of a layer identified by its index, i.e. layer_index:
    def get_kernels_output(model,layer_index,model_input,training_flag=True):
        get_outputs=K.function([model.layers[0].input,K.learning_phase()],
                                   [model.layers[layer_index].output])
        kernels_output=get_outputs([model_input,training_flag])[0]
        return kernels_output


    #define a function that combines all outputs generated by all 
    #kernels of a layer and creates a tiled image: 
    def combine_output_images(kernels_output):
        #kernels_output.shape=(10000, 36, 60, 1)

        output_count=kernels_output.shape[1]
        #edw lathos


        width=int(math.sqrt(output_count))
        height=int(math.ceil(float(output_count)/width))
        print("\n***** height:",height)
        print("\n***** width:",width)


        if len(kernels_output.shape) == 4:
            output_shape=kernels_output.shape[2:]
            image=np.zeros((height*output_shape[0],width*output_shape[0],width*output_shape[1]),
                dtype = kernels_output.dtype)

            for index,output in enumerate(kernels_output[0]):
                i=int(index/width)
                j=index%width
                print("dims:",output.shape)

                image[i*output_shape[0]:(i+1)*output_shape[0],
                      j*output_shape[1]:(j+1)*output_shape[1],0]=output
                #valueError: could not broadcast input array from 
                #shape (60,1) into shape (60,1,6)

            #provlima edw:(360, 360, 6)
            return image

    #we define a function that goes through each layer of the model 
    #and plots the tiled image generated by all kernels of that layer
    def get_model_layers_output_combined_image(model,model_input,training_flag=True):
        #training_flag:
        #False:output in test mode
        #True: output in training mode
        for layer_idx in range(len(model.layers)):
            kernels_output = get_kernels_output(model,layer_idx,model_input,training_flag)
            combined_outputs_image=combine_output_images(kernels_output)
            if combined_outputs_image is not None:
                print(model.layers[layer_idx].name)
                #stin apo katw entolh xtupaei to error
                #** shape: (6, 360, 360)
                print("image dims:",combined_outputs_image.shape)
                plt.matshow(combined_outputs_image.T,vmin=0.0,vmax=1.0)
                plt.show()

    # import TensorResponseBoard
    # tb = TensorResponseBoard.TensorResponseBoard(log_dir='../logs/', histogram_freq=10, batch_size=10,
    #                      write_graph=True, write_grads=True, write_images=True,
    #                      embeddings_freq=10,
    #                      embeddings_layer_names=['dense_1'],
    #                      embeddings_metadata='metadata.tsv',
    #                      val_size=len(x_test), 
    #                      img_path='images.jpg', 
    #                      img_size=[36, 60])


    import TensorResponseBoard

    tb=TensorResponseBoard.TensorBoardWrapper(
        final_val_gen, 
        nb_steps=5, 
        log_dir='../logs/',#self.cfg['cpdir'], 
        histogram_freq=1,
        batch_size=32, 
        write_graph=False, 
        write_grads=True
    )

    callbacks = [ModelCheckpoint('mymodel.h5',
                                monitor='val_loss',
                               save_best_only=True),
               EarlyStopping(monitor='val_loss', patience=2),
               # TensorBoard('../logs/{}', histogram_freq=10, batch_size=32,
               #             write_graph=True, write_grads=True, write_images=True,
               #             embeddings_freq=10, embeddings_metadata=None,
               #             embeddings_layer_names=embedding_layer_names),
               ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2),
               tb
               ]
    # callbacks.TensorBoard(log_dir='temp', histogram_freq=10, batch_size=32,
    #                        write_graph=True, write_grads=True, write_images=True,
    #                        embeddings_freq=10, embeddings_metadata=None,
    #                        embeddings_layer_names=embedding_layer_names)


    model.fit_generator(
        final_train_gen,
        steps_per_epoch=len(x_train) // train_batch_size,
        epochs=epochs,
        validation_data=final_val_gen,
        validation_steps=len(y_test) // test_batch_size,#,
        callbacks=callbacks
        )


    model.save('multimodal_cnn_1_epoch.h5') 

    from PIL import Image
    img_array = x_test.reshape(100, 100, 28, 28)
    img_array_flat = np.concatenate([np.concatenate([x for x in row], axis=1) for row in img_array])
    img = Image.fromarray(np.uint8(255 * (1. - img_array_flat)))
    img.save(os.path.join(log_dir, 'images.jpg'))
    np.savetxt(os.path.join(log_dir, 'metadata.tsv'), np.where(y_test)[1], fmt='%d')


