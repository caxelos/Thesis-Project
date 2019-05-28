from keras.preprocessing.image import ImageDataGenerator
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
                    yield x_train_feat[batch*train_batch_size:]
                else:
                    yield x_train_feat[batch*train_batch_size:(1+batch)*train_batch_size]

    def val_feat_gen(x_val_feat, test_batch_size):
        while True:
            for batch in range(len(x_val_feat) // test_batch_size + 1):
                if batch > max(range(len(x_val_feat) // test_batch_size)):
                    yield x_val_feat[batch*test_batch_size:]
                else:
                    yield x_val_feat[batch*test_batch_size:(1+batch)*test_batch_size]

    def merge_generator(gen1, gen2):
        while True:
            X1 = gen1.__next__()
            X2 = gen2.__next__()
            yield [X1[0], X2], X1[1]

    validation_generator = test_datagen.flow(
        x_test,
        y_test,
        batch_size=test_batch_size)

    final_train_gen = merge_generator(train_generator, train_feat_gen(x_train_feat, train_batch_size))
    final_val_gen = merge_generator(validation_generator, val_feat_gen(x_val_feat, test_batch_size))

    #callbacks = [ModelCheckpoint(MODEL_DIR+model_name+'.h5',
    #                             monitor=monitor,
    #                            save_best_only=True),
    #            EarlyStopping(monitor=monitor, patience=patience),
    #            TensorBoard(LOG_DIR+model_name+'_'+str(time())),
    #            ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2)]

    model.fit_generator(
        final_train_gen,
        steps_per_epoch=len(x_train) // train_batch_size,
        epochs=epochs,
        validation_data=final_val_gen,
        validation_steps=len(y_test) // test_batch_size)#,
        #callbacks=callbacks,)
