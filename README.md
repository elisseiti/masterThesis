# Smoke simulation for games using AI
In this project, I read vdb files from the animation using `pyopenvdb` library. 
It reads the files in the `Dust_Devil` folder. It generates a list of numpy arrays which are of different sizes. This needs reworking
The list of 3d arrays will be the input of the `generator` and the `discriminator`.

To install the `pyopenvdb` library you should use:
```sudo apt install python3-openvdb```