# General Map Making and Analysis Procedure 🔄

#### Selecting Observations
We begin by selecting a given tube slot, which in the case of this thesis would be the "i1" tube. We then make a list of all observation id's that correspond to calibration-type observations of of planets (we focus on Mars in this project) that were taken after the first alignment. We then make maps from said observations running the `make_maps_mpi.py` script, which is based on Sahanish's (I spelled that wrong) own mapping script.

For each observation, we have a map for each wafer and band combo (6 in total if all wafers are available), and we also have two additional maps for each of these combinations representing data from only left or right scan directions. In the case of the i1 tube, the available frequencies are 90Ghz and 150Ghz.

After obtaining the maps we want to use, we use the `prepare_maps.ipynb` notebook to make a dictionary to stare information about every map, which we store as `maps.json`. (We use msgspec to handle the json file to speed up load times, though all numbers must be converted to float before saving) Under the same notebook, we also classify observations as being taken during the first or second alignment (`season_class`) and during the time of day they were taken (`time_class`).

#### Fitting Models to Maps
After obtaining the dictionary of maps and checking to see what maps have existing files, we handpick one or more clean maps and test if we can fit them with a given model. This procedure is completed in the `fit_maps.ipynb` file under "Test Primary Lobe Fits". Models are stored in the `map_fun.py` file, alongside many other useful and recurring functions. New models can be built quickly by modifying an instance of the `BaseModel` class.

Once we've determined which models seem like good fit candidates, we fit all possible maps and save their results under the "Fit Models to Maps" section. Note that at this stage we don't filter bad maps, and instead assume that clean maps will have good fits, thus allowing for filtering based on fitting results. We do however make the fits using a mask around the source, which is explained in further detail in the corresponding notebook. Right after that fitting section, we use one of the models to get the center of the source and proceed to calculate the SNR.

So far, the best model we've come up with involves a 2D gaussian that allows for ellipticity, and so we mostly default to it's associated parameters when analysing maps.

#### Examining Fits

Moving forward, we only work with "clean" maps. We know from previous testing that most maps should have an ellipticity close to zero, and we also know that in most cases of interest the SNR is higher than 15. We've also confirmed through testing that an elliptical 2D gaussian should fit well the primary sidelobe of any clean map. Taking these factors into account, from this point forward we consider maps to be clean if they meet the following criteria:
- They have a valid  `EllipticGaussian` fit (besides other possible fits)
- Their SNR is 15 or higher
- Their ellipticity does not surpass 0.2
- The reduced chi2 of the Elliptical 2D fit is 5 or lower.

Note that left and right scan splits are tied to the full map itself, so we only require that the full map satisfies these conditions to be considered clean.

In the `examine_fits.ipynb` notebook we check the quality of the fits, as well as the maps that passed our "clean" filter. We take a look at the distribution of parameters, how different map categories stack together and we take careful note to check that the errors from the map maker and the fits are correctly propagated as to then use them in the map stacking procedure.

#### Fitting Residuals

After veryfing that we have good general fits for the primary lobes, we then move on to attempting to fit the residuals. Given the nature of the residuals, we're currently limited to modelling them with zernike polynomials, which despite providing good fits in a lot of cases, it does require a lot of polynomials to achieve a good fit and careful selection of the region to analyze.

A good thing about using Zernike polynomials is that we can characterise the modes that appear in the different maps, which can tell us about the defects that might be causing the residuals we see.

#### Searching Features

In the final part of this procedure, we now use all the tools and data we've gathered so far in order to search for any interesting features.