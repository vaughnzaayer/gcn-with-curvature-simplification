# Mesh Classification with Intrinsic Curvature Error Simplification Preprocessing

## Building
Start by cloning the repository and its submodules with 

```
git clone --recurse-submodules https://github.com/vaughnzaayer/gcn-with-curvature-simplification.git
```

Next, build `bulk-coarsen`:
```
cd bulk_coarsen
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
make -j4
cd ../..
```

Then, create a Python virtual environment with ``python@3.12 -m venv .venv``. Activate it with ``source .venv/bin/activate``. If you're using Linux, install the dependencies with:
```
pip install -r requirements.txt
```
If you're using MacOS, instead use:
```
pip install -r macos_requirements.txt
```

## Unpacking the Dataset
To unpack the dataset, simply run ``bash unpack_dataset.sh`` from the project's root directory.

## Preprocessing the Dataset
Initiate preprocessing with 
`./bulk_coarsen/build/bin/bulk_mesh_coarsen ./data` 

## Training the model
To start training the model, just run
```
python model.py
```
