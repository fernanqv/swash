import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.animation as animation
from bluemath_tk.interpolation.rbf import RBF


def animate_case_propagation(case_dataset, tini=0, tend=30, tstep=2, figsize=(15, 5)):
    """
    Function to animate the propagation of the swash for a single case
    """

    fig, ax = plt.subplots(1, figsize=figsize)

    # Init animation
    def init():
        return []

    # Función de actualización de la animación
    def update(frame):
        ax.clear()

        # ax.tick_params(axis="both", which="major", labelsize=plot_params["fontsize"])
        # ax.set_xlim(x_depth[0], x_depth[-1])
        # ax.set_ylim(y_depth_min[0], y_depth[-1] + 3)
        # ax.set_xlabel("Cross-shore Distance (m)", fontsize=plot_params["fontsize"])
        # ax.set_ylabel("Elevation (m)", fontsize=plot_params["fontsize"])

        # bathymetry
        # ax.fill_between(x_depth, y_depth_min, y_depth, fc="wheat", zorder=2)

        # waves
        watlev = case_dataset.isel(Tsec=frame)["Watlev"].values
        ax.fill_between(
            range(len(Y)), y_depth_min, Y, fc="deepskyblue", alpha=0.5, zorder=1
        )
        ax.set_title("Time : {0} s".format(frame), fontsize=12)

        return []

    # Crear animación
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(tini, tend, tstep), init_func=init, blit=True
    )
    plt.close()

    # Mostrar animación
    return ani


def show_graph_for_different_parameters(pp, sm, df_subset):
    # plot_PCA(xds_PCA, df_subset, lhs_parameters.get("dimensions_names"), figsize1=(6+1,6), figsize2=(10,3))

    # Function to update the plot based on widget input
    def update_plot(hs=1.5, hs_l0=0.02, vegetation=1):
        # Create dataframe
        df_dataset_single_case = pd.DataFrame(
            data={
                "Hs": [hs],
                "Hs_L0": [hs_l0],
                "VegetationHeight": [vegetation],
            }
        )

        # Spatial Reconstruction
        var = ["Hs"]  # variable to reconstruct
        X_max = 1150  # maximum spatial X to consider for PCA
        variance = 99  # maximum variance to explain

        ix_scalar_subset = [0, 1, 2]
        ix_scalar_target = [0]
        ix_directional_subset = []
        ix_directional_target = []

        # Compute PCA and apply RBF recosntruction
        xds_PCA, ds_output = RBF_Reconstruction_spatial(
            pp,
            var,
            list(df_dataset_single_case.columns),
            df_dataset_single_case,
            df_subset,
            ix_scalar_subset,
            ix_directional_subset,
            ix_scalar_target,
            ix_directional_target,
            variance,
            X_max,
        )
        ds_output_all = ds_output.assign_coords(
            vegetation=("case", df_dataset_single_case.VegetationHeight.values)
        )

        fig, ax = plt.subplots(figsize=(14, 6))
        ds_output_all["Hs"].sel(case=0).plot(x="Xp", ax=ax, color="k")
        sm.plot_depthfile(ax=ax)
        ax.plot(
            np.arange(int(pp.swash_proj.np_ini), int(pp.swash_proj.np_fin)),
            np.repeat(-2.5, int(pp.swash_proj.np_fin - pp.swash_proj.np_ini)),
            color="darkgreen",
            linewidth=int(25 * vegetation),
        )
        ax.set_ylim(-7, 4)
        ax.set_xlim(400, 1160)
        ax.grid(True)

        ax.set_title(
            f"Reconstructed Hs for Hs: {hs}, Hs_L0: {hs_l0} and VegetationHeight: {vegetation}"
        )

    # Creating widgets
    widget_hs = widgets.FloatSlider(
        value=1.5, min=0.5, max=3, step=0.5, description="Hs:"
    )
    widget_hs_l0 = widgets.FloatSlider(
        value=0.02, min=0.01, max=0.03, step=0.01, description="Hs_L0:"
    )
    widget_vegetation = widgets.FloatSlider(
        value=1, min=0, max=1.5, step=0.5, description="VegetationHeight:"
    )

    # Using interact to link widgets to the function
    return interact(
        update_plot, hs=widget_hs, hs_l0=widget_hs_l0, vegetation=widget_vegetation
    )


def show_graph_for_all_vegetations(pp, sm, df_subset, hs=2.0, hs_l0=0.02):
    # Create dataframe
    df_dataset_same_hs = pd.DataFrame(
        data={
            "Hs": np.repeat(hs, 100),
            "Hs_L0": np.repeat(hs_l0, 100),
            "VegetationHeight": np.linspace(0, 1.5, 100),
        }
    )

    # Spatial Reconstruction
    var = ["Hs"]  # variable to reconstruct
    X_max = 1150  # maximum spatial X to consider for PCA
    variance = 99  # maximum variance to explain

    ix_scalar_subset = [0, 1, 2]
    ix_scalar_target = [0]
    ix_directional_subset = []
    ix_directional_target = []

    # Compute PCA and apply RBF recosntruction
    xds_PCA, ds_output = RBF_Reconstruction_spatial(
        pp,
        var,
        list(df_dataset_same_hs.columns),
        df_dataset_same_hs,
        df_subset,
        ix_scalar_subset,
        ix_directional_subset,
        ix_scalar_target,
        ix_directional_target,
        variance,
        X_max,
    )
    ds_output_all = ds_output.assign_coords(
        vegetation=("case", df_dataset_same_hs.VegetationHeight.values)
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    # Create a colormap and a normalization based on vegetation values
    norm = colors.Normalize(
        vmin=min(ds_output_all.vegetation), vmax=max(ds_output_all.vegetation)
    )
    cmap = cm.get_cmap("YlGn", len(ds_output_all.vegetation))

    for i, case in enumerate(ds_output_all["case"].values):
        vegetation = ds_output_all["vegetation"].sel(case=case).item()
        color = cmap(norm(vegetation))
        ds_output_all["Hs"].sel(case=case).plot(
            x="Xp",
            # label=f'Case {case} (Veg={vegetation:.2f})',
            color=color,
            ax=ax,
        )

    # Add colorbar for vegetation scale
    smp = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    smp.set_array([])
    cbar = plt.colorbar(smp, ax=ax)
    cbar.set_label("Vegetation")

    sm.plot_depthfile(ax=ax)
    ax.plot(
        np.arange(int(pp.swash_proj.np_ini), int(pp.swash_proj.np_fin)),
        np.repeat(-2.5, int(pp.swash_proj.np_fin - pp.swash_proj.np_ini)),
        color="darkgreen",
        linewidth=10,
    )
    ax.set_ylim(-7, 4)
    ax.set_xlim(400, 1160)
    ax.grid(True)

    ax.set_title(
        f"Reconstructed Hs for Hs: {hs}, Hs_L0: {hs_l0} and different vegetation heights"
    )


# from lib.reconstruction import RBF_Reconstruction_singular
# from lib.output_extract import scatter_color

# # Sinular Reconstruction
# var = ['Ru2']

# ix_scalar_subset = [0, 1, 2]
# ix_scalar_target = [0]
# ix_directional_subset = []
# ix_directional_target = []

# # RBF reconstruction
# df_output = RBF_Reconstruction_singular(
#     pp, var, lhs_parameters.get("dimensions_names"),
#     df_dataset_same_hs, df_subset,
#     ix_scalar_subset, ix_directional_subset,
#     ix_scalar_target, ix_directional_target
# )

# fig = scatter_color(df_dataset_same_hs,
#                     df_output,
#                     lhs_parameters.get("dimensions_names"),
#                     var,
#                     figsize=(7,7), vmin=None, vmax=None, cmap='jet')


# g = pp.ds_output.Hs.isel(Xp=slice(0, 1150)).plot(col="case_id", col_wrap=5)
# g.fig.subplots_adjust(hspace=0.15, wspace=0.15)
# # Optional: Remove titles and axis labels if you don't want them
# # for ax in g.axes.flatten():
# #     ax.set_title('')  # Remove titles
# #     ax.set_xlabel('')  # Remove x-axis labels
# #     ax.set_ylabel('')  # Remove y-axis labels
# #     ax.tick_params(axis='both', which='both', length=0)  # Remove ticks
# # Set titles dynamically based on dataset values
# for ax, case_id in zip(g.axes.flatten(), pp.ds_output['case_id'].values):
#     ax.set_title(f"Hs: {df_subset.Hs.values[int(case_id)]:.2f} | Vegetation: {df_subset.VegetationHeight.values[int(case_id)]:.2f}")
#     # Remove the spines (edges)
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     # Retain the grid (set grid on)
#     ax.grid(True)
#     # ax.plot(np.arange(750, 1150), -sp.depth[750:1150], color="black")
