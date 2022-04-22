import numpy as np
def gen_coin_seq(s,p):
    s1, s2, s3 = s[0], s[1], s[2]
    q, r, t = p[0], p[1], p[2]
    #生成硬币序列
    np.random.seed(1)

    coin = np.random.rand(1000,100)
    coin_sum = []
    coin_class = np.random.rand(1000)
    for i in range(coin.shape[0]):
        if coin_class[i] < s1: 
            #投掷硬币为s1
            coin[i][coin[i]>(1-q)] = 1
            coin[i][coin[i]<(1-q)] = 0
        elif coin_class[i] < s1 + s2:
            #投掷硬币为s2
            coin[i][coin[i]>(1-r)] = 1
            coin[i][coin[i]<(1-r)] = 0
        else:
            #投掷硬币为s2
            coin[i][coin[i]>(1-t)] = 1
            coin[i][coin[i]<(1-t)] = 0
        coin_sum.append(coin[i].sum())
    coin_class[coin_class < s1] = 1 #a
    coin_class[(coin_class > s1) & (coin_class < (s1 + s2))] = 2 #b
    coin_class[(coin_class > (s1 + s2)) & (coin_class < 1)] = 3 #c
    # print(coin_class)
    # print(coin_sum)
    return coin

def cal_probality(coin_seq, p):
    num_p = coin_seq.sum()
    num = coin_seq.size
    probality = (p**num_p) * (1-p)**(num-num_p)
    return probality

def EM(coin,p_begin):
    q_eva, r_eva, t_eva = p_begin[0], p_begin[1], p_begin[2]
    epoch = 10
    for i in range(epoch):
        pro_list = []
        for j in range(coin.shape[0]):
            num_p = coin[j].sum()
            a_p = cal_probality(coin[j], q_eva)
            b_p = cal_probality(coin[j], r_eva)
            c_p = cal_probality(coin[j], t_eva)
            a_p_con =  a_p / (a_p+b_p+c_p)
            b_p_con =  b_p / (a_p+b_p+c_p)
            c_p_con =  c_p / (a_p+b_p+c_p)
            pro_list.append([a_p_con, b_p_con, c_p_con, num_p])
        a_p_num = 0
        a_n_num = 0
        b_p_num = 0
        b_n_num = 0
        c_p_num = 0
        c_n_num = 0
        for exp in pro_list:
            a_p_num += exp[0] * exp[3]
            a_n_num += exp[0] * (coin.shape[1]-exp[3])

            b_p_num += exp[1] * exp[3]
            b_n_num += exp[1] * (coin.shape[1]-exp[3])
            
            c_p_num += exp[2] * exp[3]
            c_n_num += exp[2] * (coin.shape[1]-exp[3])
        q_eva = a_p_num / (a_p_num + a_n_num)
        r_eva = b_p_num / (b_p_num + b_n_num)
        t_eva = c_p_num / (c_p_num + c_n_num)

        print("第%d轮迭代的概率为，a=%f,b=%f,c=%f"%(i, q_eva, r_eva, t_eva))
    s1_eva = 0 
    s2_eva = 0
    s3_eva = 0
    for exp in pro_list:
        s1_eva += exp[0] 
        s2_eva += exp[1]
        s3_eva += exp[2]
    return [q_eva, r_eva, t_eva], [s1_eva/coin.shape[0], s2_eva/coin.shape[0], s3_eva/coin.shape[0]]
if __name__ == "__main__":
    s = [0.2, 0.3, 0.5]
    p = [0.2, 0.8, 0.5]
    p_begin = [0.1, 0.7, 0.6]
    coin = gen_coin_seq(s,p)

    p_eva, s_eva = EM(coin,p_begin)
    print("真实：q=%f,r=%f,t=%f"%(p[0], p[1], p[2]))
    print("初始：q=%f,r=%f,t=%f"%(p_begin[0], p_begin[1], p_begin[2]))
    print("预测：q=%f,r=%f,t=%f"%(p_eva[0], p_eva[1], p_eva[2]))
    print("真实:s1=%f,s2=%f,s3=%f"%(s[0], s[1], s[2]))
    print("预测:s1=%f,s2=%f,s3=%f"%(s_eva[0], s_eva[1], s_eva[2]))