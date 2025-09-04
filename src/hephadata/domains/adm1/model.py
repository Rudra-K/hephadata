import yaml
import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd


class ADM1Model:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
        
        self.state_map = {
            # Soluble Substrates (Indices 0-11)
            'S_su': 0, 'S_aa': 1, 'S_fa': 2, 'S_va': 3, 'S_bu': 4,
            'S_pro': 5, 'S_ac': 6, 'S_h2': 7, 'S_ch4': 8, 'S_IC': 9,
            'S_IN': 10, 'S_I': 11,
            # Particulate Components (Indices 12-23)
            'X_xc': 12, 'X_ch': 13, 'X_pr': 14, 'X_li': 15, 'X_su': 16,
            'X_aa': 17, 'X_fa': 18, 'X_c4': 19, 'X_pro': 20, 'X_ac': 21,
            'X_h2': 22, 'X_I': 23,
            # Cations & Anions (Indices 24-25)
            'S_cation': 24, 'S_anion': 25,
            # Intermediate Physicochemical States (Indices 26-34)
            'S_H_ion': 26, 'S_va_ion': 27, 'S_bu_ion': 28, 'S_pro_ion': 29,
            'S_ac_ion': 30, 'S_hco3_ion': 31, 'S_co2': 32, 'S_nh3': 33, 'S_nh4_ion': 34,
            # Gas Phase (Indices 35-37)
            'S_gas_h2': 35, 'S_gas_ch4': 36, 'S_gas_co2': 37
        }

        self._calculate_temp_params()

    def _calculate_temp_params(self):
        p = self.params['physicochemical']
        R, T_base, T_op = p['R'], p['T_base'], p['T_op']


        self.K_w = (10**-14.0) * np.exp((p['K_w__exp_val'] / (100 * R)) * (1/T_base - 1/T_op))
        self.K_a_co2 = (10**-6.35) * np.exp((p['K_a_co2_exp_val'] / (100 * R)) * (1/T_base - 1/T_op))
        self.K_a_IN = (10**-9.25) * np.exp((p['K_a_IN_exp_val'] / (100 * R)) * (1/T_base - 1/T_op))
        self.p_gas_h2o = p['p_gas_h2o_base'] * np.exp(p['p_gas_h2o_exp_val'] * (1/T_base - 1/T_op))
        self.K_H_co2 = p['K_H_co2_base'] * np.exp((p['K_H_co2_exp_val'] / (100 * R)) * (1/T_base - 1/T_op))
        self.K_H_ch4 = p['K_H_ch4_base'] * np.exp((p['K_H_ch4_exp_val'] / (100 * R)) * (1/T_base - 1/T_op))
        self.K_H_h2 = p['K_H_h2_base'] * np.exp((p['K_H_h2_exp_val'] / (100 * R)) * (1/T_base - 1/T_op))


    def get_derivatives(self, t: float, y: np.ndarray, influent_state: np.ndarray, q_ad: float) -> np.ndarray:
        s = self.state_map
        
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I = y[s['S_su']:s['S_I']+1]
        X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I = y[s['X_xc']:s['X_I']+1]
        S_cation, S_anion = y[s['S_cation']:s['S_anion']+1]
        S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion = y[s['S_H_ion']:s['S_nh4_ion']+1]
        S_gas_h2, S_gas_ch4, S_gas_co2 = y[s['S_gas_h2']:s['S_gas_co2']+1]

        # Unpack influent state
        S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in, S_ch4_in, S_IC_in, S_IN_in, S_I_in = influent_state[s['S_su']:s['S_I']+1]
        X_xc_in, X_ch_in, X_pr_in, X_li_in, X_su_in, X_aa_in, X_fa_in, X_c4_in, X_pro_in, X_ac_in, X_h2_in, X_I_in = influent_state[s['X_xc']:s['X_I']+1]
        S_cation_in, S_anion_in = influent_state[s['S_cation']:s['S_anion']+1]


        p = self.params['biochemical']

        pH = -np.log10(S_H_ion)

        # pH inhibition for amino acid degradation
        pH_UL_aa, pH_LL_aa = p['pH_UL_aa'], p['pH_LL_aa']
        K_pH_aa = 10 ** (-(pH_LL_aa + pH_UL_aa) / 2.0)
        nn_aa = 3.0 / (pH_UL_aa - pH_LL_aa)
        I_pH_aa = (K_pH_aa ** nn_aa) / (S_H_ion ** nn_aa + K_pH_aa ** nn_aa)
        
        # pH inhibition for acetate degradation
        pH_UL_ac, pH_LL_ac = p['pH_UL_ac'], p['pH_LL_ac']
        K_pH_ac = 10 ** (-(pH_LL_ac + pH_UL_ac) / 2.0)
        n_ac = 3.0 / (pH_UL_ac - pH_LL_ac)
        I_pH_ac = (K_pH_ac ** n_ac) / (S_H_ion ** n_ac + K_pH_ac ** n_ac)
        
        # pH inhibition for hydrogen degradation
        pH_UL_h2, pH_LL_h2 = p['pH_UL_h2'], p['pH_LL_h2']
        K_pH_h2 = 10 ** (-(pH_LL_h2 + pH_UL_h2) / 2.0)
        n_h2 = 3.0 / (pH_UL_h2 - pH_LL_h2)
        I_pH_h2 = (K_pH_h2 ** n_h2) / (S_H_ion ** n_h2 + K_pH_h2 ** n_h2)
        
        # Other inhibition factors
        I_IN_lim = 1 / (1 + (p['K_S_IN'] / S_IN))
        I_h2_fa = 1 / (1 + (S_h2 / p['K_I_h2_fa']))
        I_h2_c4 = 1 / (1 + (S_h2 / p['K_I_h2_c4']))
        I_h2_pro = 1 / (1 + (S_h2 / p['K_I_h2_pro']))
        I_nh3 = 1 / (1 + (S_nh3 / p['K_I_nh3']))

        # Combine inhibition factors for process rates
        I_5 = I_pH_aa * I_IN_lim
        I_6 = I_5
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4
        I_9 = I_8
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro
        I_11 = I_pH_ac * I_IN_lim * I_nh3
        I_12 = I_pH_h2 * I_IN_lim

        # Calculate biochemical process rates (rho values)
        b = self.params['biochemical']

        # Disintegration
        rho_1 = b['k_dis'] * X_xc
        # Hydrolysis
        rho_2 = b['k_hyd_ch'] * X_ch
        rho_3 = b['k_hyd_pr'] * X_pr
        rho_4 = b['k_hyd_li'] * X_li
        # Acidogenesis / Uptake of substrates
        rho_5 = b['k_m_su'] * (S_su / (b['K_S_su'] + S_su)) * X_su * I_5
        rho_6 = b['k_m_aa'] * (S_aa / (b['K_S_aa'] + S_aa)) * X_aa * I_6
        rho_7 = b['k_m_fa'] * (S_fa / (b['K_S_fa'] + S_fa)) * X_fa * I_7
        # Acetogenesis / Uptake of VFAs
        rho_8 = b['k_m_c4'] * (S_va / (b['K_S_c4'] + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8
        rho_9 = b['k_m_c4'] * (S_bu / (b['K_S_c4'] + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9
        rho_10 = b['k_m_pro'] * (S_pro / (b['K_S_pro'] + S_pro)) * X_pro * I_10
        # Methanogenesis
        rho_11 = b['k_m_ac'] * (S_ac / (b['K_S_ac'] + S_ac)) * X_ac * I_11
        rho_12 = b['k_m_h2'] * (S_h2 / (b['K_S_h2'] + S_h2)) * X_h2 * I_12
        # Decay of biomass
        rho_13 = b['k_dec_X_su'] * X_su
        rho_14 = b['k_dec_X_aa'] * X_aa
        rho_15 = b['k_dec_X_fa'] * X_fa
        rho_16 = b['k_dec_X_c4'] * X_c4
        rho_17 = b['k_dec_X_pro'] * X_pro
        rho_18 = b['k_dec_X_ac'] * X_ac
        rho_19 = b['k_dec_X_h2'] * X_h2

        # Calculate gas transfer rates
        phys = self.params['physicochemical']
        p_gas_h2 = S_gas_h2 * phys['R'] * phys['T_op'] / 16
        p_gas_ch4 = S_gas_ch4 * phys['R'] * phys['T_op'] / 64
        p_gas_co2 = S_gas_co2 * phys['R'] * phys['T_op']
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        q_gas = phys['k_p'] * (p_gas - phys['p_atm'])
        if q_gas < 0: q_gas = 0

        rho_T_h2 = phys['k_L_a'] * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
        rho_T_ch4 = phys['k_L_a'] * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4)
        rho_T_co2 = phys['k_L_a'] * (S_co2 - self.K_H_co2 * p_gas_co2)

        # Calculate final derivatives for each state variable
        derivatives = np.zeros_like(y)
        st = self.params['stoichiometric']
        flow_term = q_ad / phys['V_liq']

        # Soluble components
        derivatives[s['S_su']] = flow_term*(S_su_in - S_su) + rho_2 + (1-st['f_fa_li'])*rho_4 - rho_5
        derivatives[s['S_aa']] = flow_term*(S_aa_in - S_aa) + rho_3 - rho_6
        derivatives[s['S_fa']] = flow_term*(S_fa_in - S_fa) + st['f_fa_li']*rho_4 - rho_7
        derivatives[s['S_va']] = flow_term*(S_va_in - S_va) + (1-st['Y_aa'])*st['f_va_aa']*rho_6 - rho_8
        derivatives[s['S_bu']] = flow_term*(S_bu_in - S_bu) + (1-st['Y_su'])*st['f_bu_su']*rho_5 + (1-st['Y_aa'])*st['f_bu_aa']*rho_6 - rho_9
        derivatives[s['S_pro']] = flow_term*(S_pro_in - S_pro) + (1-st['Y_su'])*st['f_pro_su']*rho_5 + (1-st['Y_aa'])*st['f_pro_aa']*rho_6 + (1-st['Y_c4'])*0.54*rho_8 - rho_10
        derivatives[s['S_ac']] = flow_term*(S_ac_in - S_ac) + (1-st['Y_su'])*st['f_ac_su']*rho_5 + (1-st['Y_aa'])*st['f_ac_aa']*rho_6 + (1-st['Y_fa'])*0.7*rho_7 + (1-st['Y_c4'])*0.31*rho_8 + (1-st['Y_c4'])*0.8*rho_9 + (1-st['Y_pro'])*0.57*rho_10 - rho_11
        # S_h2 is an algebraic state, solved elsewhere. Its derivative in the ODE system is 0.
        derivatives[s['S_h2']] = 0 
        derivatives[s['S_ch4']] = flow_term*(S_ch4_in - S_ch4) + (1-st['Y_ac'])*rho_11 + (1-st['Y_h2'])*rho_12 - rho_T_ch4
        # S_IC and S_IN have complex carbon and nitrogen balancing equations
        sum_rho_13_19 = rho_13+rho_14+rho_15+rho_16+rho_17+rho_18+rho_19
        c_bal = (-st['C_xc']+st['f_sI_xc']*st['C_sI']+st['f_ch_xc']*st['C_ch']+st['f_pr_xc']*st['C_pr']+st['f_li_xc']*st['C_li']+st['f_xI_xc']*st['C_xI'])*rho_1 + (-st['C_ch']+st['C_su'])*rho_2 + (-st['C_pr']+st['C_aa'])*rho_3 + (-st['C_li']+(1-st['f_fa_li'])*st['C_su']+st['f_fa_li']*st['C_fa'])*rho_4 + (-st['C_su']+(1-st['Y_su'])*(st['f_bu_su']*st['C_bu']+st['f_pro_su']*st['C_pro']+st['f_ac_su']*st['C_ac'])+st['Y_su']*st['C_bac'])*rho_5 + (-st['C_aa']+(1-st['Y_aa'])*(st['f_va_aa']*st['C_va']+st['f_bu_aa']*st['C_bu']+st['f_pro_aa']*st['C_pro']+st['f_ac_aa']*st['C_ac'])+st['Y_aa']*st['C_bac'])*rho_6 + (-st['C_fa']+(1-st['Y_fa'])*0.7*st['C_ac']+st['Y_fa']*st['C_bac'])*rho_7 + (-st['C_va']+(1-st['Y_c4'])*0.54*st['C_pro']+(1-st['Y_c4'])*0.31*st['C_ac']+st['Y_c4']*st['C_bac'])*rho_8 + (-st['C_bu']+(1-st['Y_c4'])*0.8*st['C_ac']+st['Y_c4']*st['C_bac'])*rho_9 + (-st['C_pro']+(1-st['Y_pro'])*0.57*st['C_ac']+st['Y_pro']*st['C_bac'])*rho_10 + (-st['C_ac']+(1-st['Y_ac'])*st['C_ch4']+st['Y_ac']*st['C_bac'])*rho_11 + ((1-st['Y_h2'])*st['C_ch4']+st['Y_h2']*st['C_bac'])*rho_12 + (-st['C_bac']+st['C_xc'])*sum_rho_13_19
        derivatives[s['S_IC']] = flow_term*(S_IC_in - S_IC) - c_bal - rho_T_co2
        n_bal = (st['N_xc']-st['f_xI_xc']*st['N_I']-st['f_sI_xc']*st['N_I']-st['f_pr_xc']*st['N_aa'])*rho_1 - st['Y_su']*st['N_bac']*rho_5 + (st['N_aa']-st['Y_aa']*st['N_bac'])*rho_6 - st['Y_fa']*st['N_bac']*rho_7 - st['Y_c4']*st['N_bac']*rho_8 - st['Y_c4']*st['N_bac']*rho_9 - st['Y_pro']*st['N_bac']*rho_10 - st['Y_ac']*st['N_bac']*rho_11 - st['Y_h2']*st['N_bac']*rho_12 + (st['N_bac']-st['N_xc'])*sum_rho_13_19
        derivatives[s['S_IN']] = flow_term*(S_IN_in - S_IN) + n_bal
        derivatives[s['S_I']] = flow_term*(S_I_in - S_I) + st['f_sI_xc']*rho_1

        # Particulate components
        derivatives[s['X_xc']] = flow_term*(X_xc_in - X_xc) - rho_1 + sum_rho_13_19
        derivatives[s['X_ch']] = flow_term*(X_ch_in - X_ch) + st['f_ch_xc']*rho_1 - rho_2
        derivatives[s['X_pr']] = flow_term*(X_pr_in - X_pr) + st['f_pr_xc']*rho_1 - rho_3
        derivatives[s['X_li']] = flow_term*(X_li_in - X_li) + st['f_li_xc']*rho_1 - rho_4
        derivatives[s['X_su']] = flow_term*(X_su_in - X_su) + st['Y_su']*rho_5 - rho_13
        derivatives[s['X_aa']] = flow_term*(X_aa_in - X_aa) + st['Y_aa']*rho_6 - rho_14
        derivatives[s['X_fa']] = flow_term*(X_fa_in - X_fa) + st['Y_fa']*rho_7 - rho_15
        derivatives[s['X_c4']] = flow_term*(X_c4_in - X_c4) + st['Y_c4']*rho_8 + st['Y_c4']*rho_9 - rho_16
        derivatives[s['X_pro']] = flow_term*(X_pro_in - X_pro) + st['Y_pro']*rho_10 - rho_17
        derivatives[s['X_ac']] = flow_term*(X_ac_in - X_ac) + st['Y_ac']*rho_11 - rho_18
        derivatives[s['X_h2']] = flow_term*(X_h2_in - X_h2) + st['Y_h2']*rho_12 - rho_19
        derivatives[s['X_I']] = flow_term*(X_I_in - X_I) + st['f_xI_xc']*rho_1
        
        # Cations and Anions
        derivatives[s['S_cation']] = flow_term*(S_cation_in - S_cation)
        derivatives[s['S_anion']] = flow_term*(S_anion_in - S_anion)
        
        # Gas phase
        derivatives[s['S_gas_h2']] = (-q_gas / phys['V_gas']) * S_gas_h2 + (rho_T_h2 * phys['V_liq'] / phys['V_gas'])
        derivatives[s['S_gas_ch4']] = (-q_gas / phys['V_gas']) * S_gas_ch4 + (rho_T_ch4 * phys['V_liq'] / phys['V_gas'])
        derivatives[s['S_gas_co2']] = (-q_gas / phys['V_gas']) * S_gas_co2 + (rho_T_co2 * phys['V_liq'] / phys['V_gas'])

        # Other algebraic states have a derivative of 0 in the ODE system
        derivatives[s['S_H_ion']:] = 0

        return derivatives
    
    def run_simulation(self, start_state: np.ndarray, scenario, t_span: tuple, t_eval: np.ndarray):
        print("Starting ADM1 simulation")

        def model_dynamics(t, y):
            influent = scenario.get_influent_state(t)
            flow_rate = scenario.get_flow_rate(t)
            return self.get_derivatives(t, y, influent, flow_rate)
        
        solution = solve_ivp(
            fun=model_dynamics,
            t_span=t_span,
            y0=start_state,
            t_eval=t_eval,
            method='LSODA'
        )
        print("Simulation complete:")

        results = solution.y.T

        column_name = list(self.state_map.keys())
        df = pd.DataFrame(results, columns=column_name, index=solution.t)
        df.index.name = 'time'

        return df