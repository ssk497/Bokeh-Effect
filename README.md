# Bokeh Effect
The aim of our project is to segment any image into foreground and background, for which we have used the concept of defocus map (or depth map). More deltails about the project and how the defocus map is extracted from an image is described in the project report [here](https://github.com/ssk497/Bokeh-Effect/blob/master/Bokeh_Effect.pdf).<br>

The code is written in matlab since it is already having some built-in image processing filters used in our project and we have also made use of its computer-vision library.
Matlab can be downloaded and installed from its official website ([link](https://in.mathworks.com/downloads/web_downloads)).
<br>

The project is organised as follows : 

```bash
.
├── generate_bokeh_image.m    # the main matlab script for the project (implemented by us)
│── bilateral_filter.m        # external module for 2-D bilateral filter (used in generate_bokeh_image.m)
│── get_laplacian_matrix.m    # external module implementing matting laplacian (used in generate_bokeh_image.m) 
```
