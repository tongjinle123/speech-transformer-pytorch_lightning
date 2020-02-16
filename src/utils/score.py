from ASR_metrics import utils as cal


def cal_wer(true, predict):
    return cal.calculate_wer(true, predict)


def cal_cer(true, predict):
    return cal.calculate_cer(true, predict)
