from tensorflow.python import pywrap_tensorflow
import os
import pdb
checkpoint_path = os.path.join('CKPT/song_generator/', "emo_lyrics_model-95")
# pdb.set_trace()
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)