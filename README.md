# Bokeh Effect
The aim of this project is to segment any image into foreground and background, for which the concept of defocus map is used.<br>
More details about the project and how the defocus map is extracted from an image is described in the project report [here](https://github.com/ssk497/Bokeh-Effect/blob/master/Bokeh_Effect.pdf).<br>

Team Members:
- Harshal Mittal (16116020)
- Harshit Bansal (16116021)
- Shubham Maheshwari (16116065)

### Project Structure
The project is organised as follows : 

```bash
.
├── gen_bokeh_image.py                     # main project module
│── face_detection.py                      # face detection module
│── bilateral_filter.py                    # 2D bilateral filtering module
│── closed_form_matting.py                 # external module (used for matting laplacian)
│── haarcascade_frontalface_default.xml    # external module (used in face_detection.py)

```
### Usage
The code is in python3. To generate the results:
```bash
python3 gen_bokeh_image.py <path/to/your/img/file>
