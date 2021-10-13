# COVID-XR-Predictor
Create a model &amp; build a UI to make predictions on X-Rays

## Introduction 

From https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

The repository uses the "COVID" and "Normal" folders from the above dataset. The aim was to develop a proof of concept application individual physicians would be able to bring a trained neural network to interpret X-Rays of arbitrary dimensions using a large dataset from the internet. 

## How to Use 

### Making a Model
A few models have already been trained in the /models/ folder. 

To train a new model, bring "COLAB COVID_Detection.ipynb" into a Google colab workspace and go from there.
You can either adjust the architecture in make_model(), or create your own architecture. 

Note that the input dimensions for the image must be 299x299.

### Using the Application

Run COVIDPredictorApp.py.

#### Configure Snap Area

This allows you to set the dimension of the screen that you would like to capture for analysis. Once clicied, drag the target area from the top left to the bottom right. You will know that it is successful once the X1, Y1 and X2, Y2 values have changed.

#### Model Selection

The dropdown box picks models from the /models/ folder, and allows you to select which one you'd like to make a prediction on your dataset.

#### Snap Area

Screenshots the selected area.

#### Evaluate

Outputs a prediction of the snapped area or currently loaded image.

#### Save Study

Saves the most recent screenshot and saves it to /saved_studies/

#### Load Study

This allows you to bring back studies that were saved previously. You are able to evaluate images that you have saved in the past with new models as you see fit.

## Remarks

The models that are included in this repository are fairly basic - consisting of two CNN layers followed by a dense layer. The reason for doing so is that training took quite a long time and to create a more complex model requires more resources beyond the free tier of Colab. In spite of this, I find that the results are quite good with validation accuracy near 90%. At present time, I have not tested precision nor recall unfortunately.

## Next Steps 

Ideally, I would love to add a class activation map of the input image to show where a model suspects evidence of COVID. Doing so would help clinicians identify potential areas of interest. This would go into the second box at the bottom which is currently empty.

As always, working to improve upon the model's performance is also a good idea as well. One possibility is to recreate the models that were trained in the reference papers which supposedly have very good performances (i.e. sensitivity in the 99.9%s) and add them to our cache of models. In doing so, we'd be able to easily bring their models to physicians around the world.

## References 

M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. Paper link

Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. Paper Link

