#!/usr/bin/env python
'''This script collects the latest corona case numbers from Johns Hopkins
University and displays them graphically including a fit for exponential growth'''

from datetime import date, timedelta
import warnings

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter

from scipy.optimize import curve_fit, OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)


def get_data_jhu(which_data='confirmed cases'):
    """
    Fetches Corona data from Johns Hopkins University
    Returns pd.DataFrame (index = date, column = country)

    Country specific modifications based on the UN list of countries:
    'Guernsey' and 'Jersey' are merged into 'Guernsey and Jersey'
    'Congo (Brazzaville)' and 'Republic of the Congo' are merged into 'Congo (Brazzaville)'
    'Serbia' and 'Kosovo' are merged into 'Serbia'
    'Taiwan*' is renamed to 'Taiwan'
    'occupied Palestinian territory' is renamed to 'Palestine'

    :param which_data: {'confirmed cases', 'recovered','deaths'}
    :type which_data: str
    :return: pd.DataFrame (index = datetime, column = country)
    :type return: pd.DataFrame
    """
    # github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series
    BASE_URL = r'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/' \
               r'master/csse_covid_19_data/csse_covid_19_time_series/'
    SPECIFIC_URL = {'confirmed cases': r'time_series_19-covid-Confirmed.csv',
                    'recovered': r'time_series_19-covid-Recovered.csv',
                    'deaths': r'time_series_19-covid-Deaths.csv'}

    df = pd.read_csv(BASE_URL + SPECIFIC_URL[which_data])
    df = df.drop(['Province/State', 'Lat', 'Long'], axis=1)
    df = df.rename(columns={'Country/Region': 'Country'})

    df = df.set_index('Country')
    df = df.groupby(['Country']).sum()

    # df.loc['Guernsey and Jersey'] = df.loc['Guernsey'] + df.loc['Jersey']
    df.loc['Congo (Brazzaville)'] = df.loc['Congo (Brazzaville)']
    df.loc['Serbia'] = df.loc['Serbia'] + df.loc['Kosovo']
    df = df.drop(index=[
        'Kosovo',
        'Cruise Ship',
    ])

    df = df.transpose()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('d')

    df = df.rename(columns={'The Bahamas': 'Bahamas',
                            'Taiwan*': 'Taiwan',
                            'Gambia, The': 'Gambia',
                            'occupied Palestinian territory': 'Palestine'})

    return df


def get_population():
    """
    Fetches list of countries by population (United Nations) from Wikipedia

    Makes some country-specific changes to match the country names
    with the country names in the Corona list
    :return: pd.DataFrame (index = country, columns = [Continent, Population])
    """
    URL = r'https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)'
    df = pd.read_html(URL)[3]

    df['Country or area'] = df['Country or area'].apply(lambda x: x.split('[')[0])

    df = df.drop(['UN statisticalregion[4]', 'Population(1 July 2018)', 'Change'], axis=1)

    df = df.rename(columns={'Country or area': 'Country',
                            'UN continentalregion[4]': 'Continent',
                            'Population(1 July 2019)': 'Population'})

    # Wiki lable : jhu_data_label
    labels_to_change = {'United States': 'US',
                        'South Korea': 'Korea, South',
                        'Czech Republic': 'Czechia',
                        'Ivory Coast': "Cote d'Ivoire",
                        'Vatican City': 'Holy See',
                        'DR Congo': 'Congo (Kinshasa)',
                        'Congo': 'Congo (Brazzaville)',
                        'Curaçao': 'Curacao',
                        'Réunion': 'Reunion'
                        }

    for key, value in labels_to_change.items():
        df.loc[(df['Country'] == key), 'Country'] = value

    df = df.set_index('Country')
    df = df.drop(['World'])

    return df


def normalize_to_population(df_to_norm, norm=1E6):
    """
    Normalizes country-specific data with regard to their population (default: per million)
    :param df_to_norm: Input DataFrame (colums = countries)
    :type df_to_norm: pandas.DataFrame
    :param norm: Normalization factor: e.g. 1E6 for per million
    :return: Normalized DataFrame
    """
    df_pop = get_population()
    normed_df = pd.DataFrame()

    for country in df_to_norm.columns:
        normed_df[country] = df_to_norm[country] / (df_pop.loc[country, 'Population'] / norm)

    return normed_df


def select_sort_countries(df, df_threshold, countries_to_include=[], countries_to_exclude=[],
                          threshold=100):
    """
    Selects countries according to a threshold value (considered last row).
    Two lists of countries can be passed, which should be included or excluded
    :param df: DataFrame with countries as columns
    :type df: pandas.DataFrame
    :param countries_to_include: List of countries to be considered regardless of the threshold
    :param countries_to_exclude: List of countries to be excluded regardless of the threshold
    :param threshold: Threshold value which must be exceeded
    :return: Sorted list with countries (descending according to the data value)
    """
    sorted_countries = df.sort_values(by=df.index[-1], ascending=False, axis=1).columns
    countries_above_threshold = df.columns[
        (df_threshold.iloc[-1] > threshold) & (~df.columns.isin(countries_to_exclude))]

    considered_countries = set(countries_above_threshold)
    considered_countries.update(set(countries_to_include))
    sorted_considered_countries = [y for x in sorted_countries for y in considered_countries if
                                   y == x]

    return sorted_considered_countries


def exp_growth(t, r, x_0):
    """
    Function for calculating exponential growth: f(t)=x_0 * (1+r) ** t
    :param t: time
    :param r: growth rate
    :param x_0: Initial value
    :return: f(t)=x_0 * (1+r) ** t
    """
    return x_0 * (1 + r) ** t


def extract_countries_without_case():
    """
    Compares the countries in the Johns Hopkins Corona data set with the
    countries of the world (get_population()) and stores unaffected countries
    sorted by population in 'save_countries.xlsx
    """
    population_df = get_population()
    save_countries = set(population_df.index) - set(get_data_jhu().columns)
    population_df.loc[save_countries].sort_values(by='Population', ascending=False).to_excel(
        'save_countries.xlsx')


def fit_exp_growth(series, fit_window=7, starting_date=None, forecast=14):
    """
    Fit a pd.Series with exponential growth and draw the fit curve as ax.plot()
     on the last used axis instance.
    :param series: Series to be fitted
    :type series: pd.Series
    :param fit_window: Length of the interval in days, which should be approximated
    :type fit_window: int
    :param starting_date: Start of the interval, which is to be approximated, as date
    :type starting_date: datetime.date
    :param forecast: Length of the interval to be extrapolated in days
    :type forecast: int
    """
    if starting_date:
        start_index = series.index.get_loc(starting_date)
    else:
        start_index = series.index.get_loc(series.index[-fit_window])

    xdata = list(((series.index[start_index:start_index + fit_window] - series.index[
        start_index]) / np.timedelta64(1, 'D')).astype(int))
    ydata = series.iloc[start_index:start_index + fit_window]

    try:
        popt, pcov = curve_fit(exp_growth, xdata, ydata)
    except OptimizeWarning:
        print('Error during fit of {}: Cannot be approximated with exponential growth'.format(
            series.name))
        return

    dti = pd.date_range(series.index[start_index], periods=fit_window + forecast,
                        freq='D')
    ax = plt.gca()
    ax.plot(dti, exp_growth(range(fit_window + forecast), *popt))


def set_up_fig(num_colors=10, colormap='tab10', subplots_kwargs={}):
    """
     Create a new figure
    :param num_colors: Number of unique colors
    :param colormap: colormap to apply for the color cycler
    :param subplots_kwargs: kwargs for plt.subplots e.g. figsize=(13,7)
    :return: fig, ax
    """
    fig, ax = plt.subplots(**subplots_kwargs)
    cm = plt.get_cmap(colormap)
    ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])
    return fig, ax


def set_up_axes(**kwargs):
    """
    Adjust the ticks on the axis, labeling, etc.
    :param kwargs:
        See below

        :Keyword Arguments:
        log (bool): yscale log
        ylabel (str): ylabel
        ylim (list): yrange = (ymin, ymax)
        xmin (datetime.date): xmin
        xmax (datetime.date): xmax
        y_tick_formatter (str): formatter for y ticks e.g. '%.f' or '%.e'
    """
    ax = plt.gca()

    if kwargs.get('log', True):
        plt.yscale('log')

    plt.ylabel(kwargs.get('ylabel'))

    if kwargs.get('ylim'):
        plt.ylim(kwargs.get('ylim')[0], kwargs.get('ylim')[1])

    plt.xlim(kwargs.get('xmin'), kwargs.get('xmax'))

    sundays = mdates.WeekdayLocator(byweekday=mdates.SU)
    everyday = mdates.DayLocator(interval=1)
    ax.xaxis.set_major_locator(sundays)
    ax.xaxis.set_minor_locator(everyday)

    plt.tick_params(axis='y', which='minor')

    plt.setp(ax.get_yminorticklabels()[1::2], visible=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter(kwargs.get('y_tick_formatter')))
    ax.yaxis.set_minor_formatter(FormatStrFormatter(kwargs.get('y_tick_formatter')))

    plt.grid(b=True, which='major', color='grey', linestyle='-')
    plt.grid(b=True, which='minor', color='grey', linestyle='--', linewidth=0.5)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()


def plot_timeindex_dataframe(df, countries, fit=True, **kwargs):
    """
    Creates scatter plot for alle df.columns in countries.
    :param df: Dataframe with index = DatetimeIndex and colums =countries
    :type df: pandas.DataFrame
    :param countries: Countries to be plotted from the dataframe
    :tpye countries: list
    :param fit: fit values or not
    :type fit: bool
    :param kwargs:
    See below

    :Keyword Arguments:
    fit_window (int): Length of the interval in days, which should be approximated
    fit_forecast (int): Length of the interval to be extrapolated in days
    fit_starting_date (datetime.date): Start of the interval, which is to be approximated, as date
    """
    for country in countries:
        scatter = ax.scatter(df.index, df[country])

        if fit:
            fit_exp_growth(df[country],
                           fit_window=kwargs.get('fit_window', 7),
                           forecast=kwargs.get('fit_forecast', 14),
                           starting_date=kwargs.get('fit_starting_date', None))
            ax.plot([], [], marker='o', linestyle='-', label=country,
                    color=scatter.get_facecolors()[0])
        else:
            ax.plot([], [], marker='o', label=country, color=scatter.get_facecolors()[0])


if __name__ == "__main__":
    CONF_CASES_DF = get_data_jhu()
    CONF_CASES_NORM_DF = normalize_to_population(CONF_CASES_DF)
    ACTIVE_CONF_CASES = CONF_CASES_DF - get_data_jhu('recovered')

    # extract_countries_without_case()

    fig, ax = set_up_fig(subplots_kwargs={'figsize': (42 / 2.54, 29.7 / 2.54)})

    COUNTRIES_TO_PLOT = select_sort_countries(CONF_CASES_DF, CONF_CASES_DF,
                                              countries_to_include=['Czechia',
                                                                    'Japan',
                                                                    'Austria',
                                                                    'Germany',
                                                                    'Italy',
                                                                    'Poland'],
                                              countries_to_exclude=['China',
                                                                    'France',
                                                                    'Korea, South',
                                                                    'Iran',
                                                                    'Norway',
                                                                    'Belgium',
                                                                    'Sweden',
                                                                    'Denmark'],
                                              threshold=800)

    plot_timeindex_dataframe(CONF_CASES_DF, COUNTRIES_TO_PLOT,
                             fit=True,
                             fit_window=7,
                             fit_forecast=21,
                             # fit_starting_date=date(2020, 3, 1)
                             )

    WEEKS_TO_FORECAST = 3
    SUNDAY_AFTER_NEXT_SUNDAY = date.today() + timedelta(days=(WEEKS_TO_FORECAST*7)-1 - date.today().weekday())

    set_up_axes(log=True,
                ylim=(1E1, 1E6),
                xmin=date(2020, 2, 16),
                xmax=SUNDAY_AFTER_NEXT_SUNDAY,
                ylabel='Confirmed infected persons',
                y_tick_formatter='%.f')

    # COUNTRIES_TO_PLOT = select_sort_countries(CONF_CASES_NORM_DF, CONF_CASES_DF,
    #                                           countries_to_include=['Czechia',
    #                                                                 'Japan',
    #                                                                 'Austria',
    #                                                                 'Germany',
    #                                                                 'Italy',
    #                                                                 'Poland'],
    #                                           countries_to_exclude=['China'],
    #                                           threshold=800)
    #
    # plot_timeindex_dataframe(CONF_CASES_NORM_DF, COUNTRIES_TO_PLOT,
    #                          fit=True,
    #                          fit_window=7,
    #                          fit_forecast=14,
    #                          fit_starting_date=date(2020, 2, 1)
    #                          )
    #
    # SUNDAY_AFTER_NEXT_SUNDAY = date.today() + timedelta(days=13 - date.today().weekday())
    #
    # set_up_axes(log=True,
    #             ylim=(1E0, 1E4),
    #             xmin=date(2020, 2, 16),
    #             xmax=SUNDAY_AFTER_NEXT_SUNDAY,
    #             ylabel='Bestätigte Infizierte pro 1.Mio. Einwohner',
    #             y_tick_formatter='%.f')

    plt.show()
