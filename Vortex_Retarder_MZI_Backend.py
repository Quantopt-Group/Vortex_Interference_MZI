import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, IntSlider, VBox, HBox, Layout,Checkbox, Label
import ipywidgets as widgets
import panel as pn
from bokeh.resources import INLINE
pn.extension()
widgets.Layout.description_width = '350px'  # Adjust as needed
FloatSlider.style.description_width = '350px'  # Set default for all sliders

# Backend functions (computational core)
def simulate_interference(
    qwp11_angle=45,         # λ/4 waveplate angle for Path 1 (degrees)
    qwp21_angle=45,         # λ/4 waveplate angle for Path 2 (degrees)
    qwp12_angle=45,         # λ/4 waveplate angle for Path 1 (degrees)
    qwp22_angle=45,         # λ/4 waveplate angle for Path 2 (degrees)
    hwp1_angle=0,           # HWP angle for Path 1 (degrees)
    hwp2_angle=0,           # HWP angle for Path 2 (degrees)
    theta_vr=0,            # Vortex retarder rotation angle (degrees)
    relative_phase=0,      # Relative phase between paths (radians)
    coherence_length=10,    # Coherence length (wavelengths)
    path_length_diff=0,     # Path length difference (wavelengths)
    input_qwp=True,
    use_one_hwp=False,
    use_two_hwp=False,
    use_qwp=True,

    vortex_charge=1,       # Charge of the vortex retarder (ℓ)
):
    # Coordinate grid
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)
    
    # Incident polarization (linear)
    pol_rad1 = np.radians(0)
    pol_rad2 = np.radians(90)

    def E_incident(pol_rad):
        return np.array([np.cos(pol_rad), np.sin(pol_rad)])
    
    # Jones matrices
    def J_QWP(theta):
        theta_rad = np.radians(theta)
        return np.array([
            [np.cos(theta_rad)**2 + 1j*np.sin(theta_rad)**2, (1 - 1j)*np.sin(theta_rad)*np.cos(theta_rad)],
            [(1 - 1j)*np.sin(theta_rad)*np.cos(theta_rad), np.sin(theta_rad)**2 + 1j*np.cos(theta_rad)**2]
        ])
    
    def J_HWP(theta):
        theta_rad = np.radians(theta)
        return np.array([
            [np.cos(2*theta_rad), np.sin(2*theta_rad)],
            [np.sin(2*theta_rad), -np.cos(2*theta_rad)]
        ])
    
    def J_VR(ell, theta, phi):
        if ell == 0:
            # Identity matrix (no effect for Gaussian beams)
            return np.tile(np.eye(2), (phi.shape[0], phi.shape[1], 1, 1))  # Shape (500, 500, 2, 2)
        else:
            theta_rad = np.radians(theta)
            cos_term = np.cos(ell * phi - 2 * theta_rad)
            sin_term = np.sin(ell * phi - 2 * theta_rad)
            return np.stack([
                np.stack([cos_term, -sin_term], axis=-1),
                np.stack([sin_term, cos_term], axis=-1)
            ], axis=-2)
    
    total_phase = 2 * np.pi * path_length_diff + relative_phase
    
    # Apply λ/4 waveplates
    if input_qwp:
        E1 = J_QWP(qwp11_angle) @ E_incident(pol_rad1)
        E2 = J_QWP(qwp21_angle) @ E_incident(pol_rad2)* np.exp(1j * total_phase)
    else:
        E1 = E_incident(pol_rad1)
        E2 = E_incident(pol_rad2)* np.exp(1j * total_phase)

    # Expand E1 and E2 to spatial dimensions
    E1_expanded = np.tile(E1, (X.shape[0], X.shape[1], 1))
    E2_expanded = np.tile(E2, (X.shape[0], X.shape[1], 1))
    
    # Apply vortex retarders
    J_VR1 = J_VR(vortex_charge, 0, phi)
    J_VR2 = J_VR(vortex_charge, theta_vr, phi)
    E1_vr = np.einsum('...ij,...j->...i', J_VR1, E1_expanded)
    E2_vr = np.einsum('...ij,...j->...i', J_VR2, E2_expanded)
    
    # Apply HWP to Path 2
    
    E1_qwp = np.einsum('ij,...j->...i', J_QWP(qwp12_angle), E1_vr)
    E2_qwp = np.einsum('ij,...j->...i', J_QWP(qwp22_angle), E2_vr)
    E1_hwp = np.einsum('ij,...j->...i', J_HWP(hwp1_angle), E1_vr)
    E2_hwp = np.einsum('ij,...j->...i', J_HWP(hwp2_angle), E2_vr)
    
    if use_one_hwp:
        E1_out = E1_hwp
        E2_out = E2_vr

    elif use_two_hwp:
        E1_out = E1_hwp
        E2_out = E2_hwp

    elif use_qwp:
        E1_out = E1_qwp
        E2_out = E2_qwp

    else:
        E1_out = E1_vr
        E2_out = E2_vr

    # LG amplitude profile
    def lg_amp_value(r, phi, l, w=1):
        """Amplitude profile for LG modes (Gaussian if l=0)."""
        if l == 0:
            # Pure Gaussian beam (no OAM terms)
            return np.exp(-r**2 / w**2)
        else:
            # Standard LG mode: (r/w)^|l| * e^{-r²/w²} * e^{i l ϕ}
            return (r / w)**np.abs(l) * np.exp(-r**2 / w**2) * np.exp(1j * l * phi)
        
    lg_amp = lg_amp_value(r, phi, vortex_charge)
    
    # Total field (with Gaussian visibility)
    coherence_factor = np.exp(-(path_length_diff**2) / (2 * (coherence_length**2)))
    E_total = lg_amp[..., np.newaxis] * (E1_out + coherence_factor * E2_out)
    
    intensity = np.sum(np.abs(E_total)**2, axis=-1)
    
    return intensity

def interactive_interference():
    # Widgets
    slider_style = {'description_width': '150px'}

    qwp11_angle_slider = FloatSlider(min=-90, max=90, step=0.5, value=45, description='First λ/4 Path 1 (deg)', style=slider_style)
    qwp21_angle_slider = FloatSlider(min=-90, max=90, step=0.5, value=45, description='First λ/4 Path 2 (deg)', style=slider_style)
    qwp12_angle_slider = FloatSlider(min=-90, max=90, step=0.5, value=45, description='Second λ/4 Path 1 (deg)', style=slider_style, visible=False)
    qwp22_angle_slider = FloatSlider(min=-90, max=90, step=0.5, value=45, description='Second λ/4 Path 2 (deg)', style=slider_style, visible=False)
    hwp1_angle_slider = FloatSlider(min=0, max=180, step=0.5, value=0, description='HWP 1 (deg)', style=slider_style, visible=False, layout={'display': 'none'})
    hwp2_angle_slider = FloatSlider(min=0, max=180, step=0.5, value=0, description='HWP 2 (deg)', style=slider_style, visible=False, layout={'display': 'none'})
    
    input_qwp_checkbox= Checkbox(value=True, description='Input QWP')

    one_hwp_checkbox = Checkbox(value=False, description='1 HWP')
    two_hwp_checkbox = Checkbox(value=False, description='2 HWP')
    qwp_checkbox = Checkbox(value=True, description='QWP')

    # Other sliders remain unchanged
    theta_vr = FloatSlider(min=0, max=360, step=1, value=0, description='VR Rotation (deg)', style=slider_style)
    relative_phase = FloatSlider(min=0, max=2*np.pi, step=0.1, value=0, description='Relative Phase (rad)', style=slider_style)
    vortex_charge = IntSlider(min=0, max=5, step=1, value=1, description='Vortex Charge (ℓ)', style=slider_style)
    coherence_length = IntSlider(min=1, max=10, step=1, value=10, description='Coherence length (λ)', style=slider_style)
    path_length_diff = IntSlider(min=0, max=50, step=1, value=0, description='Path length diff (λ)', style=slider_style)

    def update_waveplate_ui(change):
        # Hide all first
        qwp11_angle_slider.layout.display = 'none'
        qwp21_angle_slider.layout.display = 'none'

        qwp12_angle_slider.layout.display = 'none'
        qwp22_angle_slider.layout.display = 'none'
        hwp1_angle_slider.layout.display = 'none'
        hwp2_angle_slider.layout.display = 'none'
        
        # Show selected ones
        if input_qwp_checkbox.value:
            qwp11_angle_slider.layout.display = None
            qwp21_angle_slider.layout.display = None
        
        if one_hwp_checkbox.value:
            hwp1_angle_slider.layout.display = None
            
        elif two_hwp_checkbox.value:
            hwp1_angle_slider.layout.display = None
            hwp2_angle_slider.layout.display = None
        elif qwp_checkbox.value:
            qwp12_angle_slider.layout.display = None
            qwp22_angle_slider.layout.display = None

        # Make checkboxes mutually exclusive
        if change['owner'] == one_hwp_checkbox and change['new']:
            two_hwp_checkbox.value = False
            qwp_checkbox.value = False
        elif change['owner'] == two_hwp_checkbox and change['new']:
            one_hwp_checkbox.value = False
            qwp_checkbox.value = False
        
        elif change['owner'] == qwp_checkbox and change['new']:
            one_hwp_checkbox.value = False
            two_hwp_checkbox.value = False

    input_qwp_checkbox.observe(update_waveplate_ui, names='value')
    one_hwp_checkbox.observe(update_waveplate_ui, names='value')
    two_hwp_checkbox.observe(update_waveplate_ui, names='value')
    qwp_checkbox.observe(update_waveplate_ui, names='value')

    # # Interactive plot
    # @interact(
    #     qwp11_angle=qwp11_angle_slider,
    #     qwp21_angle=qwp21_angle_slider,
    #     qwp12_angle=qwp12_angle_slider,
    #     qwp22_angle=qwp22_angle_slider,
    #     hwp_angle=hwp_angle_slider,
    #     theta_vr=theta_vr,
    #     relative_phase=relative_phase,
    #     vortex_charge=vortex_charge,
    #     coherence_length=coherence_length,
    #     path_length_diff=path_length_diff,
    #     use_hwp=hwp_checkbox,
    #     use_qwp=qwp_checkbox
    # )
    # Create the controls column
    controls = VBox([
        Label("Input QWP"),
        input_qwp_checkbox,
        qwp11_angle_slider,
        qwp21_angle_slider,
        Label("Output Waveplates"),
        one_hwp_checkbox,
        two_hwp_checkbox,
        qwp_checkbox,
        hwp1_angle_slider,
        hwp2_angle_slider,
        qwp12_angle_slider,
        qwp22_angle_slider,
        Label("Vortex Retarder"),
        theta_vr,
        vortex_charge,
        Label("Interferometer"),
        relative_phase,    
        coherence_length,
        path_length_diff
    ])
    
    # Create an output widget for the plot
    out = widgets.Output()
    
    # Define the update function
    def update_plot(**kwargs):
        with out:
            out.clear_output(wait=True)
            intensity = simulate_interference(**kwargs)
            plt.figure(figsize=(8, 6))
            plt.imshow(intensity, cmap='viridis', extent=[-3, 3, -3, 3])
            plt.colorbar(label='Intensity')
            plt.title('LG Mode Interference with Polarization Control')
            plt.show()
    
    # Link the controls to the update function
    widgets.interactive_output(update_plot, {
        'qwp11_angle': qwp11_angle_slider,
        'qwp21_angle': qwp21_angle_slider,
        'qwp12_angle': qwp12_angle_slider,
        'qwp22_angle': qwp22_angle_slider,
        'hwp1_angle': hwp1_angle_slider,
        'hwp2_angle': hwp2_angle_slider,
        'theta_vr': theta_vr,
        'relative_phase': relative_phase,
        'vortex_charge': vortex_charge,
        'coherence_length': coherence_length,
        'path_length_diff': path_length_diff,
        'input_qwp': input_qwp_checkbox,
        'use_one_hwp': one_hwp_checkbox,
        'use_two_hwp': two_hwp_checkbox,
        'use_qwp': qwp_checkbox
    })
    
    # Display the horizontal layout
    # display(HBox([controls, out]))
    return pn.Row(controls, out)