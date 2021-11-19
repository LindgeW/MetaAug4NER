
def calc_prf(nb_right, nb_pred, nb_gold):
    p = nb_right / (nb_pred + 1e-30)
    r = nb_right / (nb_gold + 1e-30)
    f = (2 * nb_right) / (nb_gold + nb_pred + 1e-30)
    return p, r, f
