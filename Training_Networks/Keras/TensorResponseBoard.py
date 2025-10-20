import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np 
class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            #print("**\nib is:",ib,"\n**")#dict['img_input','pose_input']
            #print("**\ntb is:",tb,"\n**")#dict['gaze_output']

            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib['img_input'].shape[0],) + ib['img_input'].shape[1:]), dtype=np.float32)
                poses= np.zeros(((self.nb_steps * ib['pose_input'].shape[0],) + ib['pose_input'].shape[1:]), dtype=np.float32)

                tags = np.zeros(((self.nb_steps * tb['gaze_output'].shape[0],) + tb['gaze_output'].shape[1:]), dtype=np.uint8)
            imgs[s * ib['img_input'].shape[0]:(s + 1) * ib['img_input'].shape[0]] = ib['img_input']
            poses[s * ib['pose_input'].shape[0]:(s + 1) * ib['pose_input'].shape[0]] = ib['pose_input']
            tags[s * tb['gaze_output'].shape[0]:(s + 1) * tb['gaze_output'].shape[0]] = tb['gaze_output']
        
        self.validation_data = [ imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


######### SOLUTION 2 ############
# import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# from keras import backend as K
# from keras.models import Model
# from keras.callbacks import TensorBoard

# class TensorResponseBoard(TensorBoard):
#     def __init__(self, val_size, img_path, img_size, **kwargs):
#         super(TensorResponseBoard, self).__init__(**kwargs)
#         self.val_size = val_size
#         self.img_path = img_path
#         self.img_size = img_size

#     def set_model(self, model):
#         super(TensorResponseBoard, self).set_model(model)

#         if self.embeddings_freq and self.embeddings_layer_names:
#             embeddings = {}
#             for layer_name in self.embeddings_layer_names:
#                 # initialize tensors which will later be used in `on_epoch_end()` to
#                 # store the response values by feeding the val data through the model
#                 layer = self.model.get_layer(layer_name)
#                 output_dim = layer.output.shape[-1]
#                 response_tensor = tf.Variable(tf.zeros([self.val_size, output_dim]),
#                                               name=layer_name + '_response')
#                 embeddings[layer_name] = response_tensor

#             self.embeddings = embeddings
#             self.saver = tf.train.Saver(list(self.embeddings.values()))

#             response_outputs = [self.model.get_layer(layer_name).output
#                                 for layer_name in self.embeddings_layer_names]
#             self.response_model = Model(self.model.inputs, response_outputs)

#             config = projector.ProjectorConfig()
#             embeddings_metadata = {layer_name: self.embeddings_metadata
#                                    for layer_name in embeddings.keys()}

#             for layer_name, response_tensor in self.embeddings.items():
#                 embedding = config.embeddings.add()
#                 embedding.tensor_name = response_tensor.name

#                 # for coloring points by labels
#                 embedding.metadata_path = embeddings_metadata[layer_name]

#                 # for attaching images to the points
#                 embedding.sprite.image_path = self.img_path
#                 embedding.sprite.single_image_dim.extend(self.img_size)

#             projector.visualize_embeddings(self.writer, config)

#     def on_epoch_end(self, epoch, logs=None):
#         super(TensorResponseBoard, self).on_epoch_end(epoch, logs)

#         if self.embeddings_freq and self.embeddings_ckpt_path:
#             if epoch % self.embeddings_freq == 0:
#                 # feeding the validation data through the model
#                 val_data = self.validation_data[0]
#                 response_values = self.response_model.predict(val_data)
#                 if len(self.embeddings_layer_names) == 1:
#                     response_values = [response_values]

#                 # record the response at each layers we're monitoring
#                 response_tensors = []
#                 for layer_name in self.embeddings_layer_names:
#                     response_tensors.append(self.embeddings[layer_name])
#                 K.batch_set_value(list(zip(response_tensors, response_values)))

#                 # finally, save all tensors holding the layer responses
#                 self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)