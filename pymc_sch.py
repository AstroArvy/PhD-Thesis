import numpy as np
import pymc as pm
import arviz as az

from astropy.table import Table

# Load the fits file
data = Table.read('StellarMassesv19.fits', format='fits')

#print (data.colnames)
filter_data = data[(data['logmstar'] + np.log10(data['fluxscale']) > 9) & (data['Z'] > 0.035) & (data['Z'] < 0.065) & (data['fluxscale'] > 0.3) & (data['fluxscale'] < 3)]

fluxscale = filter_data['fluxscale']
z = filter_data['Z']
logM = filter_data['logmstar']+np.log10(fluxscale)
dlogM = filter_data['dellogmstar']

from sympy import uppergamma, N

def comoving_volume(z_initial, z_final, H0=70, Omega_M=0.3, Omega_Lambda=0.7):
    c = 3e5 #light speed in km/s as H0 is 70km/s/MPc
    # For a flat universe, Omega_k = 0
    Omega_k = 1 - Omega_M - Omega_Lambda  
    # Hubble distance in Mpc
    DH = c / H0  

    # Angular size distance
    def E(z, Omega_M=Omega_M, Omega_Lambda=Omega_Lambda, Omega_k=Omega_k):
        return np.sqrt(Omega_M * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_Lambda)

    # Integral for comoving distance DC
    def comoving_distance(z):
        return quad(lambda z: 1/E(z), 0, z)[0] * DH

    # Calculate DC for initial and final redshifts
    DC_initial = comoving_distance(z_initial)
    DC_final = comoving_distance(z_final)

    # Volume of the light cone V_C using the formula for Omega_k = 0 (flat universe)
    VC_initial = (4 * np.pi / 3) * DC_initial**3
    VC_final = (4 * np.pi / 3) * DC_final**3

    # The volume covered between z_initial and z_final is the difference
    V_360 = VC_final - VC_initial
    
    #full sky has 41253 sq degrees and GAMA has 250sq degrees
    V_gama = 143/41253 * V_360
    
    return V_gama

# Initial and final redshifts
z_initial = 1e-6
z_final = 0.065

# Calculate and print the volume of the light cone
volume = comoving_volume(z_initial, z_final)
print(f"The volume of the light cone between z={z_initial} and z={z_final} is approximately {volume:.2f} cubic Mpc.")

# Schechter function definition
def schechter_func(ln_mass, phi_star, ln_m_star, alpha):
    x = ln_mass - ln_m_star
    sch = np.log(10) * phi_star * (10 ** (x * (alpha + 1))) * np.exp(-10 ** x)
    sch = np.where(np.isfinite(sch), sch, 0)
    return sch

def schechter_int(ln_mass, phi_star, ln_m_star, alpha):
    # This function needs to be defined as per your integration requirements
    # Placeholder implementation, you should replace this with the correct integration
    return np.trapz(schechter_func(ln_mass, phi_star, ln_m_star, alpha), ln_mass)

def schechter_new(ln_mass, phi_star, ln_m_star1, alpha1, ln_m_star2, alpha2, f1):
    sch1 = schechter_func(ln_mass, phi_star, ln_m_star1, alpha1)
    sch2 = schechter_func(ln_mass, phi_star, ln_m_star2, alpha2)
    return f1 * sch1 + (1 - f1) * sch2

# Observed data


# PyMC model definition
with pm.Model() as model:
    # Priors for the parameters
    alpha1 = pm.Normal('alpha1', mu=-1, sigma=1.5)
    ln_m_star1 = pm.Normal('ln_m_star1', mu=9.7, sigma=0.005)
    alpha2 = pm.Normal('alpha2', mu=-1, sigma=1.5)
    ln_m_star2 = pm.Normal('ln_m_star2', mu=9.3, sigma=0.005)
    f1 = pm.Bound(pm.Normal, lower=0, upper=1)('f1', mu=0.5, sigma=0.1)
    
    phi_star = 32.522501276810594  # Fixed parameter
    
    # Calculate the Schechter function values
    sch = schechter_new(ln_mass_data, phi_star, ln_m_star1, alpha1, ln_m_star2, alpha2, f1)
    
    # Integrate for normalization
    norm1 = schechter_int(ln_mass_data, phi_star, ln_m_star1, alpha1)
    norm2 = schechter_int(ln_mass_data, phi_star, ln_m_star2, alpha2)
    norm = f1 * norm1 + (1 - f1) * norm2
    
    # Likelihood
    const = 1e-10  # To avoid log(0)
    likelihood = pm.Normal('likelihood', mu=sch / norm, sigma=const, observed=np.ones_like(ln_mass_data))
    
    # Inference
    trace = pm.sample(1000, tune=1000, chains=4)

# Analysis and plotting
az.plot_trace(trace)
az.summary(trace)