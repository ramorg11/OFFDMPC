# Offline Deep Model Predictive Control (MPC) for Visual Navigation
## | [Paper](https://arxiv.org/abs/2402.04797)

[Taha BOUZID](https://www.linkedin.com/in/taha-bouzid-7949431a2/)<sup></sup>,
[Youssef ALJ](https://scholar.google.fr/citations?user=it8FG0YAAAAJ&hl=en)<sup></sup>


## Abstract
In this paper, we propose a new visual navigation method
based on a single RGB perspective camera. Using the Visual Teach &
Repeat (VT&R) methodology, the robot acquires a visual trajectory consisting
of multiple subgoal images in the teaching step. In the repeat step,
we propose two network architectures, namely ViewNet and VelocityNet.
The combination of the two networks allows the robot to follow the visual
trajectory. ViewNet is trained to generate a future image based on the
current view and the velocity command. The generated future image is
combined with the subgoal image for training VelocityNet.We develop an
offline Model Predictive Control (MPC) policy within VelocityNet with
the dual goals of (1) reducing the difference between current and subgoal
images and (2) ensuring smooth trajectories by mitigating velocity
discontinuities. Offline training conserves computational resources, making
it a more suitable option for scenarios with limited computational
capabilities, such as embedded systems. We validate our experiments in
a simulation environment, demonstrating that our model can effectively
minimize the metric error between real and played trajectories.

## Keywords
visual navigation, mobile robots, MPC, control, deep learning.

## Training VuNet

### Without Storing the Loss or Plotting

To train the model without storing the loss or plotting it, use the following command:

```bash
python train_vunet.py --config config_vunet.yml
```
### Training with Storing the Loss

To train the model and store the loss values in a text file, use the following command:

```bash
python train_vunet.py --config config_vunet.yml --store_loss
```
An extra option to visualize the loss evolution at the end of the training can be used:

```bash
python train_vunet.py --config config_vunet.yml --store_loss --plot_loss
```

## Training VelocityNet

### Without Storing the Loss or Plotting

To train the model without storing the loss or plotting it, use the following command:

```bash
python train_velocitynet.py --config config_velocitynet.yml
```
Same thing can be done to store the loss and visualise it using: 

```bash
python train_velocitynet.py --config config_velocitynet.yml --store_loss --plot_loss
```

