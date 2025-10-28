# Double-Inverted-Pendulum-Swingup-MPC-CPP

## Dependencies
- libmpc++ and its dependencies (Eigen, NLopt): these can be installed by the instructions below.

## Build and install libmpc++
From the repository root:
```sh
cd libmpc
sudo ./configure.sh # to install dependencies if you don't have already
mkdir build
cd build
cmake ..
make
sudo make install
```
If any problem is encountered, refer to instructions [here](https://github.com/nicolapiccinelli/libmpc/).

## Build and Run the MPC script
From the repository root:
```sh
mkdir build
cd build
cmake ..
make
bin/double_inv_pend
```

### Plots and Animation
`plotter.py` can be used to create plot and animation after you have run the MPC script(which saves the recorded data to a csv file which in turn is used in Python for easy plotting).

![State Trajectories Plot](https://github.com/DishankJ/Double-Inverted-Pendulum-Swingup-MPC-CPP/blob/main/state_traj.png?raw=true)

![Animation](https://github.com/DishankJ/Double-Inverted-Pendulum-Swingup-MPC-CPP/blob/main/dip_anim.gif?raw=true)
