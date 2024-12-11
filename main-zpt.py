from get_XYZ import *
from utils import *
from TL_model import *
from model import *

if __name__ == '__main__':
    flight = "Flt1006"
    df_flight_path = "datasets/dataframes/df_flight.csv"
    df_flight = pd.read_csv(df_flight_path)
    xyz = get_XYZ(flight, df_flight)
    print(xyz.keys())
    print(xyz['tt'].shape)