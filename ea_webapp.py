import streamlit as st
import matplotlib.pyplot as plt
import statmorph
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Gaussian2D
from astropy.visualization import simple_norm
from astropy.modeling.models import Sersic2D
from photutils.segmentation import detect_threshold, detect_sources
import time
import statmorph
from astropy.convolution import convolve, Gaussian2DKernel
from statmorph.utils.image_diagnostics import make_figure
from scipy.stats import chisquare
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd


# Load the pickled data
with open('galaxy_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Unpack the variables
goto_morph   = data['goto_morph']
goto_mags    = data['goto_mags']
rand_morph   = data['rand_morph']
rand_mags    = data['rand_mags']
rand_morph2  = data['rand_morph2']
rand_mags2   = data['rand_mags2']

# -------------------------------
# 1. Convert to NumPy arrays
# -------------------------------
goto_morph = [np.array(x) for x in goto_morph]
goto_mags  = [np.array(x) for x in goto_mags]

rand_morph = [np.array(x) for x in rand_morph]
rand_mags  = [np.array(x) for x in rand_mags]

# -------------------------------
# 2. Construct color features
# -------------------------------
def make_color_features(mags):
    u, g, r, i, z = mags
    return np.column_stack([
        u - g,
        g - r,
        r - i,
        i - z,
        u - r
    ])

X_color_goto = make_color_features(goto_mags)
X_color_rand = make_color_features(rand_mags)

# -------------------------------
# 3. Combine morphology + color
# -------------------------------
X_goto = np.hstack([np.column_stack(goto_morph), X_color_goto])
X_rand = np.hstack([np.column_stack(rand_morph), X_color_rand])

X = np.vstack([X_goto, X_rand])
y = np.hstack([np.ones(len(X_goto)), np.zeros(len(X_rand))])

# -------------------------------
# 4. Train/test split (80/20)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Scale features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -------------------------------
# 6. Train Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# -------------------------------
# 7. Evaluate
# -------------------------------
y_pred = rf_model.predict(X_test_scaled)
probs = rf_model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, probs):.3f}")

def compute_x_y_target(ra_target, dec_target, ra_ref, dec_ref, x_ref, y_ref, 
                        ra_per_col, ra_per_row, dec_per_col, dec_per_row):
    # Compute del_ra and del_dec
    del_ra = ra_target - ra_ref
    del_dec = dec_target - dec_ref
    
    # Construct the coefficient matrix and the right-hand side vector
    A = np.array([[ra_per_col, ra_per_row],
                  [dec_per_col, dec_per_row]])
    b = np.array([del_ra, del_dec])
    
    # Solve for (x_target - x_ref) and (y_target - y_ref)
    delta = np.linalg.solve(A, b)
    
    # Compute x_target and y_target
    x_target = x_ref + delta[0]
    y_target = y_ref + delta[1]
    
    return x_target, y_target


def remove_distant_pixels(image_data, x_target, y_target, max_distance=100):
    # Get the indices of all pixels in the image
    height, width = image_data.shape[:2]
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Calculate squared distance to avoid sqrt calculation
    dist_sq = (x_indices - x_target) ** 2 + (y_indices - y_target) ** 2
    
    # Create a mask where squared distance is within the max_distance squared
    mask = dist_sq <= max_distance ** 2
    
    # Get the coordinates of the unmasked pixels
    y_unmasked, x_unmasked = np.where(mask)
    
    # Determine the bounding box of the unmasked pixels
    min_x, max_x = x_unmasked.min(), x_unmasked.max()
    min_y, max_y = y_unmasked.min(), y_unmasked.max()
    
    # Crop the image based on the bounding box
    cropped_image = image_data[min_y:max_y+1, min_x:max_x+1]
    
    # Reshape the cropped image to a 200x200 array (pad if necessary)
    new_shape = (200, 200)
    reshaped_image = np.zeros(new_shape, dtype=image_data.dtype)
    
    # Get the dimensions of the cropped image
    crop_height, crop_width = cropped_image.shape
    
    # Calculate the padding to center the cropped image into the new 200x200 array
    pad_y = (new_shape[0] - crop_height) // 2
    pad_x = (new_shape[1] - crop_width) // 2
    
    # Place the cropped image into the reshaped 200x200 array
    reshaped_image[pad_y:pad_y+crop_height, pad_x:pad_x+crop_width] = cropped_image
    
    return reshaped_image

def magnitude_to_flux(mag, zero_point=25.0):
    """Convert magnitude to flux assuming a zero-point magnitude."""
    return 10**((zero_point - mag) / 2.5)

def create_gaussian_psf(size=25, fwhm=3, magnitude=18.23993, zero_point=25.0):
    """Generate a Gaussian PSF with a given magnitude."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to standard deviation
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x0, y0 = size // 2, size // 2  # Center of the PSF
    
    gaussian = Gaussian2D(amplitude=1.0, x_mean=x0, y_mean=y0, x_stddev=sigma, y_stddev=sigma)
    psf = gaussian(x, y)
    
    # Normalize the PSF to sum to 1
    psf /= np.sum(psf)
    
    # Scale to desired flux
    total_flux = magnitude_to_flux(magnitude, zero_point)
    psf *= total_flux
    
    return psf

def plot_psf(psf):
    """Plot the PSF."""
    norm = simple_norm(psf, 'log')
    plt.imshow(psf, norm=norm, origin='lower', cmap='inferno')
    plt.colorbar(label='Flux')
    plt.title('Gaussian PSF')
    plt.show()

def get_morph(ra, dec):
    try:
        pos = coords.SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        # Get r-band science image (corrected frame)
        images = SDSS.get_images(coordinates=pos, band='r')
        header = images[0][0].header
        image_data = images[0][0].data

        # Reference pixel info from header
        x_ref = header['CRPIX1']
        y_ref = header['CRPIX2']
        ra_ref = header['CRVAL1']
        dec_ref = header['CRVAL2']
        ra_per_col = header['CD1_1']
        ra_per_row = header['CD1_2']
        dec_per_col = header['CD2_1']
        dec_per_row = header['CD2_2']

        # Convert RA/DEC to pixel coordinates
        x_target, y_target = compute_x_y_target(
            ra_target=ra, dec_target=dec,
            ra_ref=ra_ref, dec_ref=dec_ref,
            x_ref=x_ref, y_ref=y_ref,
            ra_per_col=ra_per_col, ra_per_row=ra_per_row,
            dec_per_col=dec_per_col, dec_per_row=dec_per_row
        )

        # Trim image around target galaxy
        image_data = remove_distant_pixels(
            image_data=image_data,
            x_target=x_target,
            y_target=y_target
        )

        # Create synthetic PSF
        psf = create_gaussian_psf(size=25, fwhm=3, magnitude=18.5)

        # Ensure float format for convolution
        image_data = image_data.astype('float64')
        threshold = detect_threshold(image_data, nsigma=3)
        convolved_image = convolve(image_data, psf)

        # Detect sources and create segmentation map
        segmap = detect_sources(convolved_image, threshold, npixels=4)
        if not segmap.segments:
            return "No sources detected"

        # Select largest object by size
        object_sizes = [seg.data.size for seg in segmap.segments]
        largest_index = int(np.argmax(object_sizes))

        # Infer gain from filename
        gain_map = [4.71, 4.6, 4.72, 4.76, 4.725, 4.895]

        # Run statmorph
        morph_list = statmorph.source_morphology(
            image_data, segmap, gain=gain_map[header['CAMCOL']-1], psf=psf
        )

        target_morph = morph_list[largest_index]
        if target_morph.flag == 0:
            return target_morph
        else:
            return f"Bad fit for object! Flag = {target_morph.flag}"

    except FileNotFoundError:
        return f"Object Not Found"
    except Exception as e:
        return f"Error processing: {e}"
    

def eaProbability(
    ra,
    dec,
    morph,
    scaler,
    output_csv=None,
    ra_list=None, dec_list=None,
    return_numpy=False
):
    """
    Rank new galaxies by E+A probability and optionally save to CSV or return as NumPy array.

    Parameters:
    - morph_batch: list of 7 arrays (Conc, Asym, Gini, M20, Smooth, Ellip, Sersic_n)
    - mag_batch:   list of 5 arrays (u, g, r, i, z)
    - scaler:      fitted StandardScaler
    - model:       trained RandomForestClassifier
    - output_csv:  optional string, filename to save ranked results (default: None)
    - ra_list:     optional list/array of RA values
    - dec_list:    optional list/array of Dec values
    - return_numpy: if True, return as NumPy array instead of DataFrame

    Returns:
    - DataFrame or NumPy array, sorted by E+A probability (descending)
    """
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # Input RA/DEC
    ra = 116.5535
    dec = 26.9230
    target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')


    # Query SDSS PhotoObj within 2 arcsec
    results = SDSS.query_region(target_coord,
                                radius=2*u.arcsec,
                                photoobj_fields=['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'type'],
                                spectro=False)

    # Filter to galaxies only (type = 3)
    galaxies = results[results['type'] == 3]

    if len(galaxies) > 0:
        # Find the closest galaxy
        coords = SkyCoord(ra=galaxies['ra']*u.deg, dec=galaxies['dec']*u.deg)
        separations = target_coord.separation(coords)
        idx_closest = separations.argmin()

        mag_batch = [
                galaxies['u'][idx_closest],
                galaxies['g'][idx_closest],
                galaxies['r'][idx_closest],
                galaxies['i'][idx_closest],
                galaxies['z'][idx_closest]
            ]

    
    morph_batch = [morph.concentration, morph.asymmetry, morph.gini, morph.m20, morph.smoothness, morph.ellipticity_centroid, morph.sersic_n]
    scaler = scaler
    model = rf_model

    # Construct feature matrix
    X_morph = np.column_stack(morph_batch)
    u, g, r, i, z = mag_batch
    X_color = np.column_stack([
        u - g,
        g - r,
        r - i,
        i - z,
        u - r
    ])
    X_all = np.hstack([X_morph, X_color])
    X_scaled = scaler.transform(X_all)

    # Predict probabilities
    probs = model.predict_proba(X_scaled)[:, 1]

    # Build DataFrame
    df = pd.DataFrame(X_all, columns=[
        'Conc', 'Asym', 'Gini', 'M20', 'Smooth', 'Ellip', 'Sersic_n',
        'u-g', 'g-r', 'r-i', 'i-z', 'u-r'
    ])
    df['prob_E+A'] = probs

    if ra_list is not None and dec_list is not None:
        df['RA'] = np.array(ra_list)
        df['Dec'] = np.array(dec_list)

    df_sorted = df.sort_values('prob_E+A', ascending=False)

    # Save to CSV if requested
    if output_csv:
        df_sorted.to_csv(output_csv, index=False)
        print(f"Saved ranked candidates to: {output_csv}")

    return df_sorted.to_numpy() if return_numpy else df_sorted






st.title("E+A Galaxy Classifier")

# User input
ra = st.number_input("Right Ascension (RA)", value=180.0)
dec = st.number_input("Declination (DEC)", value=0.0)

# Run analysis
if st.button("Analyze Galaxy"):
    try:
        morph = get_morph(ra, dec)
        
        if isinstance(morph, str):
            st.error(morph)
        else:
            # Display E+A probability
            prob = eaProbability(ra=ra,dec=dec,morph=morph,scaler=scaler,return_numpy=True)
            prob_scalar = prob[0, -1]  # Last element in the 1x13 array
            st.success(f"E+A Probability: {prob_scalar:.2f}")       


            # Display morphology figure
            fig = make_figure(morph)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")
