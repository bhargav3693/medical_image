import h5py
import json
import os
import sys

def fix_keras_h5(filepath):
    print(f"Fixing {filepath}...")
    with h5py.File(filepath, 'r+') as f:
        if 'model_config' in f.attrs:
            model_config = f.attrs['model_config']
            config_str = model_config.decode('utf-8') if isinstance(model_config, bytes) else model_config
            new_config_str = config_str.replace('/', '_')
            f.attrs['model_config'] = new_config_str.encode('utf-8')
            print("Fixed model_config")
            
        # Also need to fix layer names in the root attributes if any
        if 'layer_names' in f.attrs:
            layer_names = f.attrs['layer_names']
            new_layer_names = [n.decode('utf-8').replace('/', '_').encode('utf-8') if isinstance(n, bytes) else n.replace('/', '_') for n in layer_names]
            f.attrs['layer_names'] = new_layer_names
            print("Fixed layer_names")

        # Recursively update all group and dataset names? H5 allows renaming, but it's tricky.
        # Actually usually just model_config and layer_names are sufficient because Keras 3 loads by name and looks up weights by name, 
        # but wait, if Keras looks up weights by the layer name, we MUST rename the groups in HDF5 as well, or we can just set safe_mode=False and compile=False
        # Let's try to just adjust model_config and layer_names initially. Wait! In older h5py, model weights are stored in groups named after the layer.
        # If we change layer name in config, Keras will look for the group `conv1_conv`, but in H5 it's `conv1/conv`.

if __name__ == "__main__":
    fix_keras_h5(sys.argv[1])
