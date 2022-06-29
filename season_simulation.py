import copy

import numpy as np
import pandas as pd

from config import LEAGUE_CONFIG
from numpy.random import poisson


def load_538_dataset():
    """
    Function to load football data using the fivethirtyeight api

    :return:
    df: The dataframe containing the fivethirtyeight data
    """

    # Set the url where the 538 football data is hosted
    data_url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches.csv'

    # Load in the data
    df = pd.read_csv(data_url)


    # Extract only the relevant columns
    df = df[['season',
             'date',
             'league',
             'team1',
             'team2',
             'score1',
             'score2',
             'xg1',
             'xg2',
             'nsxg1',
             'nsxg2']]

    return df


def extract_season_data(df, league, year):
    """
    Function which extracts data for a given season from the
    fivethirtyeight data

    :param df: The dataframe containing the fivethirtyeight data
    :param league: The league to extract data for
    :param year: The year to extract data for

    :return:
    prior_df: Extracted data for the season so far
    post_df: Extracted data for un-played games in the season
    """

    # Extract the full league name from config
    league_title = LEAGUE_CONFIG[league]

    # Extract data only for required season
    season_df = df[df.league == league_title]
    season_df = season_df[season_df.season == year].copy()

    # Split out the prior and post dataframes
    prior_df = copy.deepcopy(season_df.dropna())
    post_df = copy.deepcopy(season_df[season_df.score1.isna()])[['team1', 'team2']]

    return prior_df, post_df


def simulate_season(prior_df, post_df, n_sims=10000):
    """
    Function which simulates a season many times to understand where each
    team is expected to finish at the end of the season

    :param prior_df: The dataframe containing games which have already been played
    :param post_df: The dataframe containing games yet to be played in the season
    :param n_sims: The number of distinct simulations to perform

    :return:
    position_df: A dataframe showing the probability distribution of final positions
    for each team
    """

    # Calculate an aggregated score for each match, using G, xG & nsxG
    prior_df['agg1'] = (prior_df.score1 + prior_df.xg1 + prior_df.nsxg1) / 3
    prior_df['agg2'] = (prior_df.score2 + prior_df.xg2 + prior_df.nsxg2) / 3

    # Sum the aggregated scores for the home team
    home_df = prior_df.groupby('team1').agg(
        for_sum=('agg1', 'sum'),
        away_sum=('agg2', 'sum'),
        count=('agg1', 'count')).reset_index()

    home_df.columns = ['team', 'for', 'agg', 'count']

    # Sum the aggregated scores for the away team
    away_df = prior_df.groupby('team2').agg(
        for_sum=('agg2', 'sum'),
        away_sum=('agg1', 'sum'),
        count=('agg1', 'count')).reset_index()

    away_df.columns = ['team', 'for', 'agg', 'count']

    # Join the home and away dataframes, then sum those
    stats_df = pd.concat([home_df, away_df])
    stats_df = stats_df.groupby('team')['for', 'agg', 'count'].sum().reset_index()

    # Calculate the for/against aggregated score per game for each team
    stats_df['att'] = stats_df['for'] / stats_df['count']
    stats_df['def'] = stats_df['agg'] / stats_df['count']

    # Only keep required columns
    stats_df = stats_df[['team', 'att', 'def']]

    # Join the team ratings onto the upcoming fixtures dataframe
    post_df = post_df.merge(stats_df, left_on='team1', right_on='team')
    post_df = post_df.merge(stats_df, left_on='team2', right_on='team')

    # Clean the dataframe
    post_df = post_df.drop(['team_x', 'team_y'], axis=1)
    post_df.columns = ['team1', 'team2', 'att1', 'def1', 'att2', 'def2']

    # Calculated the average aggregated score for home and away
    avgH = prior_df[['score1', 'xg1', 'nsxg1']].values.mean()
    avgA = prior_df[['score2', 'xg2', 'nsxg2']].values.mean()
    avg = prior_df[['score1', 'xg1', 'nsxg1', 'score2', 'xg2', 'nsxg2']].values.mean()

    # Calculate how many goals each team is expected to score in the upcoming games
    post_df['exp1'] = post_df.att1 * post_df.def2 * avgH / (avg ** 2)
    post_df['exp2'] = post_df.att2 * post_df.def1 * avgA / (avg ** 2)

    # Create a simulation dataframe, formed out of numerous copies of post_df
    simulation_df = pd.concat([post_df] * n_sims)

    # Define a simulation ID column
    simulation_df['simulation'] = np.repeat(range(1, n_sims + 1), len(post_df))

    # Simulate a score for each team in each simulation
    simulation_df['score1'] = poisson(simulation_df.exp1)
    simulation_df['score2'] = poisson(simulation_df.exp2)

    # Keep only relevant columns
    simulation_df = simulation_df[['team1', 'team2', 'score1', 'score2', 'simulation']]

    # Take a copy of existing results
    non_simulation_df = prior_df[['team1', 'team2', 'score1', 'score2']]

    # Create a df made from numerous copies of existing results
    non_simulation_df = pd.concat([non_simulation_df] * n_sims)
    non_simulation_df['simulation'] = np.repeat(range(1, n_sims + 1), len(prior_df))

    # Join the existing results to the simulated results
    results_df = pd.concat([simulation_df, non_simulation_df])

    # Calculate the points scored by each team in every game
    results_df['points1'] = 3 * (results_df.score1 > results_df.score2) + (results_df.score1 == results_df.score2)
    results_df['points2'] = 3 * (results_df.score2 > results_df.score1) + (results_df.score1 == results_df.score2)

    # Calculate the home and away points and gF/gA in each simulation
    results_home = results_df.groupby(['team1', 'simulation'])['score1', 'score2', 'points1'].sum().reset_index()
    results_away = results_df.groupby(['team2', 'simulation'])['score2', 'score1', 'points2'].sum().reset_index()

    # Rename the columns
    results_home.columns = ['team', 'simulation', 'for', 'agg', 'points']
    results_away.columns = ['team', 'simulation', 'for', 'agg', 'points']

    # Join the home and away results into a league table dataframe
    table_df = pd.concat([results_home, results_away])
    table_df = table_df.groupby(['simulation', 'team'])['for', 'agg', 'points'].sum().reset_index()

    # Calculate goal difference
    table_df['diff'] = table_df['for'] - table_df['agg']

    # Extract data about teams in the league
    teams = table_df.team.unique()
    num_teams = len(teams)

    # Create the league rankings for each team in every simulation
    table_df = table_df.sort_values(['simulation', 'points', 'diff', 'for'], ascending=[True, False, False, False])
    table_df['position'] = np.tile(range(1, num_teams + 1), n_sims)

    # Create a dataframe which will store data about how frequently a team finishes in a certain position
    position_df = pd.DataFrame({'team': np.repeat(teams, num_teams),
                                'position': np.tile(range(1, num_teams + 1), num_teams)})

    # Define an empty probability column
    position_df['prob'] = np.nan

    # Loop through all teams and work out how often the finish in a certain position
    for ix, row in position_df.iterrows():
        prob = len(table_df[(table_df.team == row.team) & (table_df.position == row.position)]) / n_sims

        position_df.at[ix, 'prob'] = prob

    # Pivot the table so that it isn't in long format
    position_df = pd.pivot_table(position_df, values='prob', index='team', columns='position')

    return position_df


if __name__ == '__main__':
    df = load_538_dataset()

    prior_df, post_df = extract_season_data(df, 'B1', 2022)
    position_df = simulate_season(prior_df, post_df)
