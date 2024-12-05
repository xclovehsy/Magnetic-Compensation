from get_XYZ import *
from utils import *
from TL_model import *

if __name__ == '__main__':
    flight = "Flt1006"
    df_flight_path = "datasets/dataframes/df_flight.csv"
    df_flight = pd.read_csv(df_flight_path)

    xyz = get_XYZ(flight, df_flight)
    print(xyz.keys())

    map_name = "Eastern_395"  # select map, full list in df_map
    df_options = [55770.0, 56609.0]
    line = "1006.08"  # select flight line (row) from df_options
    ind = get_ind(xyz, tt_lim=[df_options[0], df_options[1]])  # get Boolean indices
    print("ind:55770.0-56609.0")

    TL_i = 5
    df_cal_path = "datasets/dataframes/df_cal.csv"
    df_cal = pd.read_csv(df_cal_path)
    TL_ind = get_ind(xyz, tt_lim=[df_cal['t_start'][TL_i], df_cal['t_end'][TL_i]])
    print("TL_ind:{}-{}".format(df_cal['t_start'][TL_i], df_cal['t_end'][TL_i]))

    # p1 = plot_mag(xyz, ind=ind, use_mags=['mag_1_uc', 'mag_4_uc', 'mag_5_uc'], detrend_data=True)
    # p2 = plot_mag(xyz, ind=ind, use_mags=['flux_d'], detrend_data=True, vec_terms=True)
    # p2.show()

    # lpf = get_bpf(pass1=0.0, pass2=0.2, fs=10.0)
    # lpf_sig = -bpf_data(xyz['cur_strb'][TL_ind], bpf=lpf)  # apply low-pass filter, sign switched for easier comparison
    # plt.figure()
    # plt.plot(xyz['tt'][TL_ind], xyz['cur_strb'][TL_ind])
    # plt.figure()
    # plt.plot(xyz['tt'][TL_ind], lpf_sig)
    # plt.show()

    lambd = 0.025  # ridge parameter for ridge regression
    use_vec = "flux_d"  # selected vector (flux) magnetometer
    use_sca = "mag_4_uc"
    terms_A = ["permanent", "induced", "eddy"]  # Tolles-Lawson terms to use
    Bx = xyz.get(use_vec + '_x')  # load Flux D data
    By = xyz.get(use_vec + '_y')
    Bz = xyz.get(use_vec + '_z')
    Bt = xyz.get(use_vec + '_t')
    TL_d_4 = create_TL_coef(Bx[TL_ind], By[TL_ind], Bz[TL_ind], Bt[TL_ind], xyz.get(use_sca)[TL_ind], lambd=lambd,
                            terms=terms_A)  # coefficients with Flux D & Mag 4
    # print(TL_d_4)
    A = create_TL_A(Bx[ind], By[ind], Bz[ind], Bt=Bt[ind])  # Tolles-Lawson `A` matrix for Flux D
    mag_1_sgl = xyz['mag_1_c'][ind]  # professionally compensated tail stinger, Mag 1
    mag_4_uc = xyz['mag_4_uc'][ind]  # uncompensated Mag 4
    mag_4_c = mag_4_uc - detrend(np.dot(A, TL_d_4), type='linear')  # compensated Mag 4
    tt = (xyz['tt'][ind] - xyz['tt'][ind][1]) / 60
    plot(tt, mag_1_sgl, detrend_data=True, detrend_type="linear")
    plot(tt, mag_4_uc, detrend_data=True, detrend_type="linear")
    plot(tt, mag_4_c, detrend_data=True, detrend_type="linear")
    plt.show()

    # TODO NN+TL train/test data split
    # df_all_path = "datasets/dataframes/df_all.csv"
    # df_all = pd.read_csv(df_all_path)
    # df_options = df_all[df_all['flight'] == flight]

    # ind  = get_ind(xyz,line,df_options); # get Boolean indices
    # flts = [:Flt1003,:Flt1004,:Flt1005,:Flt1006] # select flights for training
    # df_train = df_all[(df_all.flight .∈ (flts,) ) .& # use all flight data
    #                   (df_all.line   .!= line),:]    # except held-out line
    # lines_train = df_train.line # training lines
