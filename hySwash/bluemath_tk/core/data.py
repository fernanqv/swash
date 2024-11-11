import logging
from typing import Tuple
import pandas as pd


def normalize(
    data: pd.DataFrame, custom_scale_factor: dict = {}, logger: logging.Logger = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize data to 0-1 using min max scaler approach

    Parameters
    ----------
    data : pd.DataFrame
        Input data to be normalized.
    custom_scale_factor : dict, optional
        Dictionary with variables as keys and a list with two values as
        values. The first value is the minimum and the second value is the
        maximum used to normalize the variable. If not provided, the
        minimum and maximum values of the variable are used.
    logger : logging.Logger, optional
        Logger object to log warnings if the custom min or max is bigger or
        lower than the datapoints.

    Returns
    -------
    normalized_data : pd.DataFrame
        Normalized data.

    scale_factor : dict
        Dictionary with variables as keys and a list with two values as
        values. The first value is the minimum and the second value is the
        maximum used to normalize the variable.

    Notes
    -----
    - This method does not modify the input data, it creates a copy of the
      dataframe and normalizes it.
    - The normalization is done variable by variable, i.e. the minimum and
      maximum values are calculated for each variable.
    - If custom min or max is bigger or lower than the datapoints, it will
      be changed to the minimum or maximum of the datapoints and a warning
      will be logged.

    Examples
    --------
    >>> import pandas as pd
    >>> from bluemath_tk.core.data import normalize
    >>> df = pd.DataFrame(
    ...     {
    ...         "Hs": np.random.rand(1000) * 7,
    ...         "Tp": np.random.rand(1000) * 20,
    ...         "Dir": np.random.rand(1000) * 360,
    ...     }
    ... )
    >>> normalized_data, scale_factor = normalize(data=data)
    """

    normalized_data = data.copy()  # Copy pd.DataFrame to avoid bad memory replacements
    scale_factor = (
        custom_scale_factor.copy()
    )  # Copy dict to avoid bad memory replacements
    for data_var in normalized_data.columns:
        data_var_min = normalized_data[data_var].min()
        data_var_max = normalized_data[data_var].max()
        if custom_scale_factor.get(data_var):
            if custom_scale_factor.get(data_var)[0] > data_var_min:
                if logger is not None:
                    logger.warning(
                        f"Proposed min custom scaler for {data_var} is bigger than datapoint, using smallest datapoint"
                    )
                scale_factor[data_var][0] = data_var_min
            else:
                data_var_min = custom_scale_factor.get(data_var)[0]
            if custom_scale_factor.get(data_var)[1] < data_var_max:
                if logger is not None:
                    logger.warning(
                        f"Proposed max custom scaler for {data_var} is lower than datapoint, using biggest datapoint"
                    )
                scale_factor[data_var][1] = data_var_max
            else:
                data_var_max = custom_scale_factor.get(data_var)[1]
        else:
            scale_factor[data_var] = [data_var_min, data_var_max]
        normalized_data[data_var] = (normalized_data[data_var] - data_var_min) / (
            data_var_max - data_var_min
        )
    return normalized_data, scale_factor


def denormalize(normalized_data: pd.DataFrame, scale_factor: dict) -> pd.DataFrame:
    """
    Denormalize data using provided scale_factor.

    Parameters
    ----------
    normalized_data : pd.DataFrame
        Input data that has been normalized and needs to be denormalized.
    scale_factor : dict
        Dictionary with variables as keys and a list with two values as
        values. The first value is the minimum and the second value is the
        maximum used to denormalize the variable.

    Returns
    -------
    pd.DataFrame
        Denormalized data.

    Notes
    -----
    - This method does not modify the input data, it creates a copy of the
      dataframe and denormalizes it.
    - The denormalization is done variable by variable, i.e. the minimum and
      maximum values are used to scale the data back to its original range.
    - Assumes that the scale_factor dictionary contains appropriate min and
      max values for each variable in the normalized_data.

    Examples
    --------
    >>> import pandas as pd
    >>> from bluemath_tk.core.data import denormalize
    >>> df = pd.DataFrame(
    ...     {
    ...         "Hs": np.random.rand(1000),
    ...         "Tp": np.random.rand(1000),
    ...         "Dir": np.random.rand(1000),
    ...     }
    ... )
    >>> scale_factor = {
    ...     "Hs": [0, 7],
    ...     "Tp": [0, 20],
    ...     "Dir": [0, 360],
    ... }
    >>> denormalized_data = denormalize(normalized_data=df, scale_factor=scale_factor)
    """

    data = normalized_data.copy()  # Copy pd.DataFrame to avoid bad memory replacements
    for data_var in data.columns:
        data[data_var] = (
            data[data_var] * (scale_factor[data_var][1] - scale_factor[data_var][0])
            + scale_factor[data_var][0]
        )
    return data
