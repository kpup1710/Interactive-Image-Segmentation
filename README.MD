# <div align="center">Pixta Interactive Segmentation</div>

## <div align="center">Introduction</div>

The source is used to support segment annotation from prior provided by user interaction. The repo now just used for local. Main function supporting annotations in this repo:
- Auto create blobs for active interaction (watershed)
- Segment main objects within a square selection (Grabcut) (Not done yet)
- Add/remove selection based on click (Not done yet)


## <div align="center">Quick Start Tutorial</div>

Run file `interactive_segment.py` for interactive segment.

```
python interactive_segment.py --img_path {path/to/img} --img_size {input image size}
```

Example

```
python interactive_segment.py --img_path source/images/kitten.jpg --img_size 640 640
```

**How to use**
- Blobs:
  - Click blob if you want to segment that bob
  - Click segmented blob if you want to remove segment annotation for that blob

- Extreme points:
  - First download model pretrained
  ```
  cd lib/supporters/extreme_points/models
  ./download_dextr_model.sh
  cd ../../../..
  ```
  
- Before start, make sure you have downloaded weights for pointwise technique from [here](https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h18_itermask.pth) and put it in the weights folder inside supporters/model


| Button                 |                    Function                           |
|------------------------|-------------------------------------------------------|
|**General options**     |                                                       |
|  'q'                   |  Decrease the mouse's radius                          |
|  'e'                   |  Increase the mouse's radius                          |
|  's'                   |  Save segment output                                  |
|  'm'                   |  Merge current mask with final output                 |
|  'r'                   |  Replace final output with current mask               |
|  'p'                   |  Clear current mask                                   |
|  esc                   |  Exit the app                                         |
|                        |                                                       |
|**Technique selections**|                                                       |
|  '1'                   |  Scribble                                             |
|  '2'                   |  Blobs                                                |
|  '3'                   |  Grabcut                                              |
|  '4'                   |  Extreme points                                       |
|  '5'                   |  Pointwise                                            |
|                        |                                                       |
|**Scribble options**    |                                                       |
|  '`'                   |  Change from draw positive to negative (or otherwise) |
|                        |                                                       |
|**Blobs options**       |                                                       |
|  'a'                   |  Change level for blob function (down)                |
|  'd'                   |  Change level for blob function (up)                  |
|                        |                                                       |
|**Grabcut options**     |                                                       |
|  '`'                   |  Change from box drawing to scribble (or otherwise)   |
|  'n'                   |  Start grabcut algorithm (after draw rectangle)       |
|  'z'                   |  background drawing                                   |   
|  'x'                   |  foreground drawing                                   |
|  'c'                   |  probable background drawing                          |
|  'v'                   |  Sprobable foreground drawing                         |
|                        |                                                       |
|**Pointwise options**   |                                                       |
|  'h'                   | Increase probability threshold                        |
|  'n'                   | Decrease probability threshold                        |
|  'f'                   | Finish object (use when segmenting many objects)      |
|  'u'                   | Undo click                                            |

**Create new class for support interactive segmentation**

Create new class in supporter file with following func

```python
class newSupporter:
    def __init__(self,):
        """Initial variables setup"""
        pass

    def mouse_cb(self, event, x, y, flags, **kwargs):
        """ What you do with each mouse event during annotation

            event: event of mouse (left button down, left button up, ...)
            x, y: coordinate of the mouse
            **kwargs: What you want to use from main class annotator
        """
        pass

    def process(self, in_button):
        """ What you do with each button you select

            in_button: input button
        """
        pass

    def reset_env_activate(self, img):
        """ What you want to do when reset the environment 
            of the supporter if you change supporter in annotator or 
            change level of the supporter
        """
        pass

    def get_instruction_image(self, ins_image = None):
        """For visualize instruction"""
        pass
```