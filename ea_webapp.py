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


import joblib

rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')


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
    height, width = image_data.shape[:2]
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    dist_sq = (x_indices - x_target) ** 2 + (y_indices - y_target) ** 2
    mask = dist_sq <= max_distance ** 2

    y_unmasked, x_unmasked = np.where(mask)

    if len(x_unmasked) == 0 or len(y_unmasked) == 0:
        raise ValueError("No unmasked pixels found within trimming radius.")

    min_x, max_x = x_unmasked.min(), x_unmasked.max()
    min_y, max_y = y_unmasked.min(), y_unmasked.max()

    cropped_image = image_data[min_y:max_y+1, min_x:max_x+1]

    new_shape = (200, 200)
    reshaped_image = np.zeros(new_shape, dtype=image_data.dtype)
    crop_height, crop_width = cropped_image.shape
    pad_y = (new_shape[0] - crop_height) // 2
    pad_x = (new_shape[1] - crop_width) // 2
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
    Predict E+A probability for a galaxy using morphology + ugriz mags from SDSS.
    """
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Build SkyCoord
    target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    try:
        # Query SDSS photometry within 2 arcsec
        results = SDSS.query_region(
            target_coord,
            radius=2*u.arcsec,
            photoobj_fields=[
                'ra', 'dec',
                'modelMag_u', 'modelMag_g', 'modelMag_r',
                'modelMag_i', 'modelMag_z',
                'type'
            ],
            spectro=False
        )

        # DEBUG: Log SDSS query result
        try:
            with open("sdss_debug_log.txt", "a") as f:
                f.write(f"RA={ra:.5f}, Dec={dec:.5f}\n")
                f.write(f"SDSS query returned: {results}\n\n")
        except Exception as e:
            print(f"Failed to write debug log: {e}")


        if results is None or len(results) == 0:
            return "No SDSS photometric object found near this position"

        # Filter to galaxies (type = 3)
        galaxies = results[results['type'] == 3]
        if len(galaxies) == 0:
            return "No SDSS galaxy found (type=3) within 2 arcsec"

        # Find closest galaxy
        coords = SkyCoord(ra=galaxies['ra']*u.deg, dec=galaxies['dec']*u.deg)
        separations = target_coord.separation(coords)
        idx_closest = separations.argmin()

        # Get ugriz model magnitudes
        mag_batch = [
            galaxies['modelMag_u'][idx_closest],
            galaxies['modelMag_g'][idx_closest],
            galaxies['modelMag_r'][idx_closest],
            galaxies['modelMag_i'][idx_closest],
            galaxies['modelMag_z'][idx_closest]
        ]

    except Exception as e:
        return f"SDSS query error: {e}"

    try:
        # Morphology vector
        morph_batch = [
            morph.concentration,
            morph.asymmetry,
            morph.gini,
            morph.m20,
            morph.smoothness,
            morph.ellipticity_centroid,
            morph.sersic_n
        ]

        # Construct full feature vector
        X_morph = np.array(morph_batch).reshape(1, -1)
        u, g, r, i, z = mag_batch
        X_color = np.array([[u - g, g - r, r - i, i - z, u - r]])
        X_all = np.hstack([X_morph, X_color])

        # Scale and predict
        X_scaled = scaler.transform(X_all)
        probs = rf_model.predict_proba(X_scaled)[:, 1]

        # Build result DataFrame
        df = pd.DataFrame(X_all, columns=[
            'Conc', 'Asym', 'Gini', 'M20', 'Smooth', 'Ellip', 'Sersic_n',
            'u-g', 'g-r', 'r-i', 'i-z', 'u-r'
        ])
        df['prob_E+A'] = probs

        if ra_list is not None and dec_list is not None:
            df['RA'] = np.array(ra_list)
            df['Dec'] = np.array(dec_list)

        df_sorted = df.sort_values('prob_E+A', ascending=False)

        if output_csv:
            df_sorted.to_csv(output_csv, index=False)
            print(f"Saved ranked candidates to: {output_csv}")

        return df_sorted.to_numpy() if return_numpy else df_sorted

    except Exception as e:
        return f"Prediction error: {e}"

def get_and_plot_spectrum(ra, dec):
    from astroquery.sdss import SDSS
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    try:
        xid = SDSS.query_region(pos, radius=2*u.arcsec, spectro=True)
        if xid is None or len(xid) == 0:
            return "No spectra found near this position"

        # Find nearest spectrum
        sp = SDSS.get_spectra(matches=xid[:1])[0]
        spec_data = sp[1].data
        flux = spec_data['flux']
        loglam = spec_data['loglam']
        wavelength = 10**loglam

        # Plot the spectrum
        fig, ax = plt.subplots()
        ax.plot(wavelength, flux, color='black', lw=0.5)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.set_title("SDSS Spectrum")
        ax.set_xlim(wavelength[0], wavelength[-1])
        return fig

    except Exception as e:
        return f"Error retrieving spectrum: {e}"




st.title("E+A Galaxy Classifier")

# User input
ra = st.number_input("Right Ascension (RA)", value=180.0, format="%.5f")
dec = st.number_input("Declination (DEC)", value=0.0, format="%.5f")


# Run analysis
if st.button("Analyze Galaxy"):
    try:
        morph = get_morph(ra, dec)
        
        if isinstance(morph, str):
            st.error(morph)
        else:
            # Display E+A probability
            with st.spinner("Calculating E+A probability..."):
                prob = eaProbability(ra=ra, dec=dec, morph=morph, scaler=scaler, return_numpy=True)

            prob_scalar = prob[0, -1]
            st.success(f"E+A Probability: {prob_scalar}")
     


            # Display morphology figure
            fig = make_figure(morph)
            st.pyplot(fig)

            #Display spectrum if available
            with st.spinner("Retrieving SDSS spectra..."):
                spectrum_fig = get_and_plot_spectrum(ra, dec)

            if isinstance(spectrum_fig, str):
                st.warning(spectrum_fig)
            else:
                st.pyplot(spectrum_fig)


    except Exception as e:
        st.error(f"An error occurred: {e}")

# ========================= new stuff begins (simpler Excel with images) =========================
import io, os, tempfile
from pathlib import Path

st.header("Batch Analysis")

uploaded_file = st.file_uploader("Upload CSV with 'RA' and 'DEC' columns", type=['csv'])

if uploaded_file is not None:
    try:
        df_coords = pd.read_csv(uploaded_file)
        if 'RA' not in df_coords.columns or 'DEC' not in df_coords.columns:
            st.error("CSV must contain 'RA' and 'DEC' columns.")
        else:
            batch_results = []
            total = len(df_coords)

            # temp folder for all per-run PNGs
            workdir = Path(tempfile.mkdtemp(prefix="ea_batch_"))
            spec_dir = workdir / "spectra"
            morph_dir = workdir / "morph"
            spec_dir.mkdir(parents=True, exist_ok=True)
            morph_dir.mkdir(parents=True, exist_ok=True)

            with st.spinner(f"Processing {total} coordinates..."):
                for i, row in df_coords.iterrows():
                    ra_val = float(row['RA'])
                    dec_val = float(row['DEC'])
                    st.write(f"🔍 Processing {i+1}/{total}: RA={ra_val}, DEC={dec_val}")

                    # Run your morphology
                    morph = get_morph(ra_val, dec_val)

                    # Prepare a result row (start with RA/DEC)
                    if isinstance(morph, str):
                        result = {"RA": ra_val, "DEC": dec_val, "Error": morph}
                        morph_fig_path = ""
                    else:
                        # Model prediction (your function already returns a one-row DF)
                        prob_df = eaProbability(ra_val, dec_val, morph, scaler)
                        if isinstance(prob_df, str):
                            result = {"RA": ra_val, "DEC": dec_val, "Error": prob_df}
                        else:
                            result = prob_df.iloc[0].to_dict()
                            result["RA"] = ra_val
                            result["DEC"] = dec_val
                            # Save statmorph figure -> PNG
                            try:
                                fig_m = make_figure(morph)
                                morph_fig_path = str(morph_dir / f"statmorph_{i}.png")
                                fig_m.savefig(morph_fig_path, dpi=150, bbox_inches="tight")
                                plt.close(fig_m)
                            except Exception:
                                morph_fig_path = ""

                    # Try to render + save SDSS spectrum PNG (best-effort)
                    try:
                        spec_fig = get_and_plot_spectrum(ra_val, dec_val)
                        if hasattr(spec_fig, "savefig"):
                            spec_fig_path = str(spec_dir / f"spectrum_{i}.png")
                            spec_fig.savefig(spec_fig_path, dpi=150, bbox_inches="tight")
                            plt.close(spec_fig)
                        else:
                            spec_fig_path = ""
                    except Exception:
                        spec_fig_path = ""

                    # Attach image paths (used for embedding in Excel)
                    result["SDSS_spectrum_img"] = spec_fig_path
                    result["statmorph_img"]     = morph_fig_path

                    batch_results.append(result)

            results_df = pd.DataFrame(batch_results)

            st.success("Batch processing complete!")
            st.dataframe(results_df)

            # -------- Write Excel with embedded thumbnails --------
            import xlsxwriter

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                sheet = "results"
                results_df.to_excel(writer, sheet_name=sheet, index=False)

                wb = writer.book
                ws = writer.sheets[sheet]

                # widen columns and set row heights for small thumbs
                for c, name in enumerate(results_df.columns):
                    ws.set_column(c, c, max(14, min(40, len(str(name)) + 2)))
                row_height = 120  # px
                img_scale = 0.5   # downscale images to fit
                for r in range(len(results_df)):
                    ws.set_row(r + 1, row_height)  # +1 for header

                # figure out the two image columns
                col_spec  = results_df.columns.get_loc("SDSS_spectrum_img")
                col_morph = results_df.columns.get_loc("statmorph_img")

                for r in range(len(results_df)):
                    # spectrum image
                    p = results_df.iat[r, col_spec]
                    if isinstance(p, str) and p and os.path.exists(p):
                        ws.insert_image(r + 1, col_spec, p, {"x_scale": img_scale, "y_scale": img_scale, "object_position": 1})
                    # statmorph image
                    p = results_df.iat[r, col_morph]
                    if isinstance(p, str) and p and os.path.exists(p):
                        ws.insert_image(r + 1, col_morph, p, {"x_scale": img_scale, "y_scale": img_scale, "object_position": 1})

            xlsx_bytes = output.getvalue()
            st.download_button(
                label="📥 Download Results as Excel (.xlsx) with images",
                data=xlsx_bytes,
                file_name="ea_predictions_batch_with_images.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            # ------------------------------------------------------

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
# ========================= new stuff ends =========================


st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 0.9em;'>"
    "Made by Jacob Yuzovitskiy. Contact: "
    "<a href='mailto:jacob.yuzovitskiy@macaulay.cuny.edu'>jacob.yuzovitskiy@macaulay.cuny.edu</a>"
    "</div>",
    unsafe_allow_html=True
)

