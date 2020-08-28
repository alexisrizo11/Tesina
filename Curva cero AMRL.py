
import numpy as np
from numpy import nan as Nan
import pandas as pd
import math
import scipy.optimize as sp
from scipy.interpolate import interp1d
from xbbg import blp
import blpapi as bbg
import pdblp
con = pdblp.BCon(debug=False,timeout=100000).start()
from datetime import date as dt
import datetime
import os

path_curva_spot = 'Ruta donde se busca poner los datos resultantes de la cuva cupón cero'


def path_dia(path_inicial,fecha):
    path_dia = path_inicial + fecha + '/'
    return path_dia

def path_final(path_con_fecha,economia):
    path_final = path_con_fecha + economia + '/'
    return path_final

def path_final_final(path_final,metodo):
    path = path_final + metodo + '/'
    return path

dict_eco_dias = {
    'Canada' : 365,
    'MXN' : 360
}

dict_economia_FaceAmount = {
    'Canada' : 1000000,
    'MXN' : 100000
}


def cusips_con_corp(cusips):
    cusips_corp = []
    for i in cusips:
        cusips_corp.append(i + ' Corp')
    return cusips_corp

def bdp(cusips):
    datos = blp.bdp(cusips,['SECURITY_NAME','MATURITY'])
    return datos

def settle_dt(cusips,fecha):
    datos = blp.bdp(cusips,['SETTLE_DT'],USER_LOCAL_TRADE_DATE=fecha)
    return datos

def bdh(cusips,fecha):
    datos = blp.bdh(cusips,['PX_DIRTY_MID','YLD_YTM_MID'],fecha,fecha)
    return datos


def datos_instrumentos(cusips,fecha,datos_bdp,settle_dt,datos_bdh):
    dic_cusip_PrecioTasa = {}
    for cusip in cusips:
        dic_cusip_PrecioTasa[cusip] = (datos_bdh.iloc[0][(cusip,'PX_DIRTY_MID')],datos_bdh.iloc[0][(cusip,'YLD_YTM_MID')])
    
    datos_bdh_convertidos = pd.DataFrame(dic_cusip_PrecioTasa)
    datos_bdh_convertidos =  datos_bdh_convertidos.transpose()
    datos_bdh_convertidos.reset_index(inplace=True)
    datos_bdh_convertidos.rename(columns={'index':'ticker',0: "Precio_Sucio", 1: "YTM"},inplace=True)
    datos_bdp.reset_index(inplace=True)
    datos_completos = pd.merge(datos_bdp,datos_bdh_convertidos,on='ticker')
    settle_dt.reset_index(inplace=True)
    datos_completos = pd.merge(datos_completos,settle_dt,on='ticker')
    datos_completos['dias_a_vencimiento'] = (datos_completos['maturity'] - datos_completos['settle_dt']).dt.days
    datos_completos['años'] = datos_completos['dias_a_vencimiento']/360
    datos_completos['YTM/100'] = datos_completos['YTM']/100
    datos_completos['ytm_continua'] = np.log(1+datos_completos['YTM/100'])
    datos_completos.sort_values(by=['dias_a_vencimiento'],inplace=True)
    datos_completos.reset_index(inplace=True)
    datos_completos.drop('index',axis=1,inplace=True)
    indice = list(range(1,len(datos_completos.index)+1))
    indice = pd.Series(indice)
    datos_completos['indice'] = indice
    datos_completos.set_index('indice',inplace=True)
    datos_completos.loc[0] = [Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan]
    datos_completos.loc[-1] = [Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan]
    datos_completos.sort_index(inplace=True)
    datos_completos.loc[0,'años'] = 1 / 360

    return datos_completos


def cash_flows(cusips,fecha):

    dic_cusip_Cashflows = {}
    for cusip in cusips:
        des_cash_flow = blp.bds(cusip,"DES_CASH_FLOW", USER_LOCAL_TRADE_DATE=fecha)
        des_cash_flow['Flujo_Total'] = des_cash_flow['coupon_amount'] + des_cash_flow['principal_amount']
        des_cash_flow['factor_descuento'] = 0
        dic_cusip_Cashflows[cusip] = des_cash_flow

    return dic_cusip_Cashflows


def fechas_pagos(cusips,cash_flows):
    fechas_pagos = []
    for cusip in cusips:
        for fecha in cash_flows[cusip]['payment_date']:
            if fecha not in fechas_pagos:
                fechas_pagos.append(fecha)
            else:
                ()
    fechas_pagos.sort()
    return fechas_pagos


def fd(rt):
    tasas_forward_discretas = rt.copy()
    tasas_forward_discretas = tasas_forward_discretas[['ticker','años']]
    tasas_forward_discretas['fd'] = np.nan
    tasas_forward_discretas.loc[1,'fd'] = rt.loc[1,'r']
    for i in range(2,len(tasas_forward_discretas.index)):
        fd = ((rt.loc[i,'r']*rt.loc[i,'años']-rt.loc[i-1,'r']*rt.loc[i-1,'años'])/(rt.loc[i,'años']-rt.loc[i-1,'años']))
        tasas_forward_discretas.loc[i,'fd'] = fd
    return tasas_forward_discretas

def f(fd):
    tasas_forward = fd.copy()
    tasas_forward['f'] = np.nan
    tasas_forward.loc[0,'años'] = 0
    n = len(tasas_forward.index)-1
    for i in range(1,n):
        f = tasas_forward.loc[i+1,'fd']*((tasas_forward.loc[i,'años']-tasas_forward.loc[i-1,'años'])/(tasas_forward.loc[i+1,'años']-tasas_forward.loc[i-1,'años']))+ tasas_forward.loc[i,'fd']*((tasas_forward.loc[i+1,'años']-tasas_forward.loc[i,'años'])/(tasas_forward.loc[i+1,'años']-tasas_forward.loc[i-1,'años']))
        if f > 3*min(tasas_forward.loc[i+1,'fd'],tasas_forward.loc[i,'fd']):
            f = 2*min(tasas_forward.loc[i+1,'fd'],tasas_forward.loc[i,'fd'])
        else:
            ()
        tasas_forward.loc[i,'f'] = f
    f_0 = tasas_forward.loc[1,'fd']-0.5*(tasas_forward.loc[1,'f']-tasas_forward.loc[1,'fd'])
    if f_0 > 3*min(tasas_forward.loc[1,'fd'],tasas_forward.loc[0,'fd']):
        f = 2*min(tasas_forward.loc[1,'fd'],tasas_forward.loc[0,'fd'])
    else:
        ()
    tasas_forward.loc[0,'f'] = f_0
    tasas_forward.loc[n,'f'] = tasas_forward.loc[n,'fd']-0.5*(tasas_forward.loc[n-1,'f']-tasas_forward.loc[n,'fd'])
    tasas_forward.drop('fd',axis=1,inplace=True)
    return tasas_forward

def r(tasa_referencia,cusips_mm,cusips_instrumentos,datos_instrumentos):

    columnas = ['ticker','años','r']
    indice = list(range(0,len(cusips_instrumentos)+2))
    rt = pd.DataFrame(index=indice,columns=columnas)
    rt.loc[1,'ticker'] = 'Tasa_Referencia'
    rt.loc[1,'años'] = 1/360
    rt.loc[1,'r'] = tasa_referencia
    for i in range(2,len(cusips_mm)+2):
        rt.loc[i,'ticker'] = datos_instrumentos.loc[i-1,'ticker']
        rt.loc[i,'años'] = datos_instrumentos.loc[i-1,'años']
        rt.loc[i,'r'] = datos_instrumentos.loc[i-1,'ytm_continua']

    return rt


def ultimo_indice(t,datos_instrumentos):
    n = datos_instrumentos.index[-1]
    if t<0:
        ultimo_indice = 0

    elif t > datos_instrumentos.loc[n,'años']:
         ultimo_indice = n

    else:
        for i in datos_instrumentos.index:
            
            if datos_instrumentos.loc[i,'años'] == t:
                ultimo_indice = i
                return ultimo_indice

            else:
                if ((t > datos_instrumentos.loc[i,'años']) & (t < datos_instrumentos.loc[i+1,'años'])):
                    ultimo_indice = i
                    return ultimo_indice
                else:
                    ()
    return ultimo_indice

def G_mc(t,t_anterior,t_siguiente,x,g_0,g_1):

    if ((x == 0) | (x == 1) | ((g_0 == 0) & (g_1 == 0))):
        G = 0

    elif (((g_0 < 0) & (g_1 >= g_0*(-0.5)) & (g_1 <= g_0*(-2))) | ((g_0 > 0) & (g_1 <= g_0*(-0.5)) & (g_1 >= g_0*(-2)))):
        G = (t_siguiente - t_anterior) * (g_0 * (x - 2*x**2 + x**3) + g_1 * (-x**2 + x**3))

    elif (((g_0 < 0) & (g_1 > g_0*(-2))) | ((g_0 > 0) & (g_1 < g_0*(-2)))):
        eta = (g_1 + 2*g_0) / (g_1 - g_0)
        if x <= eta:
            G = g_0 * (t - t_anterior)
        else:
            G = g_0 * (t - t_anterior) + (g_1 - g_0) * ((x - eta)**3) / ((1 - eta)**2) / 3 * (t_siguiente - t_anterior)

    elif (((g_0 > 0) & (g_1 < 0) & (g_1 > g_0*(-0.5))) | ((g_0 < 0) & (g_1 > 0) & (g_1 < g_0*(-0.5)))):
        eta = 3 * g_1 / (g_1 - g_0)
        if x < eta:
            G = (t_siguiente - t_anterior) * (g_1*x - 1/3*(g_0-g_1)*(((eta-x)**3)/(eta**2)-eta))
        else:
            G = (t_siguiente - t_anterior) * ((2/3*g_1* + 1/3*g_0) * eta + g_1 * (x-eta))
            
    else:
        eta = g_1 / (g_0 + g_1)
        A = (-g_0 * g_1) / (g_0 + g_1)
        if x <= eta:
            G = (t_siguiente - t_anterior) * (A*x - 1/3 * (g_0-A) * (((eta-x)**3)/(eta**2)-eta))
        else:
            G = (t_siguiente - t_anterior) * ((2/3*A + 1/3*g_0) * eta + (A*(x-eta) + (g_1-A)/3*((x-eta)**3)/((1-eta)**2)))
    return G



def mc(t,rt,f,fd):
    
    #Indices
    n = rt.index[-1]
    indice_anterior = ultimo_indice(t,rt)
    indice_siguiente = indice_anterior + 1
    
    #t(i)
    t_1 = rt.loc[1,'años']
    t_n = rt.loc[n,'años']
    try:
        ti_anterior = rt.loc[indice_anterior,'años']
    except:
        ()
    try:
        ti_siguiente = rt.loc[indice_siguiente,'años']
    except:
        ()
    
    
    #x(t)
    try:
        x_t = (t-ti_anterior)/(ti_siguiente-ti_anterior)
    except:
        ()
    
    
    #fd(i)
    fd_0 = fd.loc[0,'fd']
    try:
        fd_anterior = fd.loc[indice_anterior,'fd']
    except:
        ()
    
    try:
        fd_siguiente = fd.loc[indice_siguiente,'fd']
    except:
        ()
    
    
    #f(i)
    f_0 = f.loc[0,'f']
    try:
        f_anterior = f.loc[indice_anterior,'f']
    except:
        ()
    
    try:
        f_siguiente = f.loc[indice_siguiente,'f']
    except:
        ()
    
    f_n = f.loc[n,'f']
    
    #g(x)
    try:
        g_0 = f_anterior - fd_siguiente
    except:
        ()
    
    try:
        g_1 = f_siguiente - fd_siguiente
    except:
        ()
    
    
    #r(t)
    try:
        r_anterior = rt.loc[indice_anterior,'r']
    except:
        ()
    
    r_n = rt.loc[n,'r']
    
    if t < t_1:
        mc = f_0
    elif t >= t_n:
        mc = (1 / t) * (r_n * t_n + f_n * (t - t_n))
    else:
        G = G_mc(t,ti_anterior,ti_siguiente,x_t,g_0,g_1)
        
        mc = (1/t) * (r_anterior*ti_anterior + fd_siguiente * (t-ti_anterior) + G)

    return mc


def tasa_spot(t,rt,f,fd,metodo):
    if metodo == 'monotone_convex':
        tasa_spot = mc(t=t,rt=rt,f=f,fd=fd)

    else:
        ()

    return tasa_spot


def bootstrapp(economia,tasa_referencia,cus_mm,cus_bonos,metodo,fecha,iteraciones):
    
    #Datos economía
    dias_totales = dict_eco_dias[economia]
    FaceAmount = dict_economia_FaceAmount[economia]

    #Construcción cusips
    cusips_mm = cusips_con_corp(cusips=cus_mm)
    cusips_bonos = cusips_con_corp(cusips=cus_bonos)
    cusips_instrumentos = cusips_mm + cusips_bonos
    
    #Datos Instrumentos
    datos_bdp_instrumentos = bdp(cusips=cusips_instrumentos)
    settle_instrumentos = settle_dt(cusips=cusips_instrumentos,fecha=fecha)
    st_dt = settle_instrumentos.iloc[-1] ['settle_dt']
    #st_dt = settle_instrumentos.loc[cusips_bonos[0],'settle_dt']
    datos_bdh_instrumentos = bdh(cusips=cusips_instrumentos,fecha=fecha)
    datos_instrum = datos_instrumentos(cusips=cusips_instrumentos,fecha=fecha,datos_bdp=datos_bdp_instrumentos,settle_dt=settle_instrumentos,datos_bdh=datos_bdh_instrumentos)
    
    #Datos cash flows bonos
    CF = cash_flows(cusips=cusips_bonos,fecha=fecha)
    fechas_CF = fechas_pagos(cusips=cusips_bonos,cash_flows=CF)
    venc = datos_instrum['maturity'].drop(0)
    venc = list(venc)
    fechas_pagos_ExVencimiento = list(set(fechas_CF) - set(venc))

    #Conversión tasa de referencia
    tasa_referencia_continua = np.log(1+tasa_referencia/100)
    
    tasas_nodos = {}
    diferencias_tasas_nodos = {}

    #Algoritmo para encontrar Tasas spot en vencimientos

    # ( 1 ) Supón rn
    r_supuesta = r(tasa_referencia=tasa_referencia_continua,cusips_mm=cus_mm,cusips_instrumentos=cusips_instrumentos,datos_instrumentos=datos_instrum)
    for i in range(len(cusips_mm)+2,len(cusips_instrumentos)+2):
        r_supuesta.loc[i,'ticker'] = datos_instrum.loc[i-1,'ticker']
        r_supuesta.loc[i,'años'] = datos_instrum.loc[i-1,'años']
        r_supuesta.loc[i,'r'] = datos_instrum.loc[i-1,'ytm_continua']
        
    # ( 2 ) Ajusta el método de interpolación con las tasas del paso 1 (Ajusta la interpolación a las fechas de pago de cupón r_1,r_2,...,r_n-1)
    contador = 1

    while contador <= iteraciones:
        
        
        
        if contador == 1:

            columnas = ['años','tasa_spot']
            indice = fechas_pagos_ExVencimiento
            spot_fechas_pagos = pd.DataFrame(index=indice,columns=columnas)

            fd_1ra_iteracion = fd(rt=r_supuesta)
            f_1ra_iteracion = f(fd=fd_1ra_iteracion)

            for t  in fechas_pagos_ExVencimiento:
                vencimiento_años = ((t - st_dt).days) / dias_totales
                r_spot = tasa_spot(t=vencimiento_años,rt=r_supuesta,f=f_1ra_iteracion,fd=fd_1ra_iteracion,metodo=metodo)
                spot_fechas_pagos.loc[t,'años'] = vencimiento_años
                spot_fechas_pagos.loc[t,'tasa_spot'] = r_spot

            # ( 3 ) Encuentra rn
            r_iteracion = r(tasa_referencia=tasa_referencia_continua,cusips_mm=cus_mm,cusips_instrumentos=cusips_instrumentos,datos_instrumentos=datos_instrum)
            datos_bonos = datos_instrum.copy()
            datos_bonos.reset_index(inplace=True)
            datos_bonos.set_index('ticker',inplace=True)
            for bono in cusips_bonos:
                print('bono es ' + str(bono))
                matriz = CF[bono]
                matriz.reset_index(inplace=True)
                matriz.set_index('payment_date',inplace=True)
                indice_matriz_sin_vencimiento = matriz.index
                indice_matriz_sin_vencimiento = list(indice_matriz_sin_vencimiento)
                indice_matriz_sin_vencimiento.pop(-1)
                #Primeras iteraciones
                for i in indice_matriz_sin_vencimiento:
                    matriz.loc[i,'factor_descuento'] = math.exp(-spot_fechas_pagos.loc[i,'tasa_spot']*spot_fechas_pagos.loc[i,'años'])
                matriz['DCF'] = matriz['Flujo_Total'] * matriz['factor_descuento']
                precio_bono_teorico = matriz['DCF'].sum()
                precio_teorico_sin_ultimo_pago = precio_bono_teorico - matriz.iloc[-1]['DCF']
                precio_bono_mkt = datos_bonos.loc[bono,'Precio_Sucio'] * FaceAmount / 100
                ultimo_pago = matriz.iloc[-1]['Flujo_Total']
                fecha_ultimo_pago = matriz.index[-1]
                def buscar_objetivo_DF(rn):
                    DF = ((precio_bono_mkt - precio_teorico_sin_ultimo_pago) / ultimo_pago) - rn
                    return DF
                df_n =  sp.broyden1(buscar_objetivo_DF,1)
                rn = math.log1p(1/df_n-1)
                años_ultimo_pago = ((fecha_ultimo_pago - st_dt).days) / 360
                rn = rn/ años_ultimo_pago
                spot_fechas_pagos.loc[fecha_ultimo_pago,'tasa_spot'] = rn
                spot_fechas_pagos.loc[fecha_ultimo_pago,'años'] = años_ultimo_pago
                posicion_bono = len(cusips_mm) + cusips_bonos.index(bono) + 2
                r_iteracion.loc[posicion_bono,'ticker'] = bono
                r_iteracion.loc[posicion_bono,'años'] = datos_bonos.loc[bono,'años']
                r_iteracion.loc[posicion_bono,'r'] = rn
                tasas_nodos[bono] = rn
                
            contador = contador + 1

        else:

            columnas = ['años','tasa_spot']
            indice = fechas_pagos_ExVencimiento
            spot_fechas_pagos = pd.DataFrame(index=indice,columns=columnas)

            fd_iteracion = fd(rt=r_iteracion)
            f_iteracion = f(fd=fd_iteracion)

            for t  in fechas_pagos_ExVencimiento:
                vencimiento_años = ((t - st_dt).days) / dias_totales
                r_spot = tasa_spot(t=vencimiento_años,rt=r_iteracion,f=f_iteracion,fd=fd_iteracion,metodo=metodo)
                spot_fechas_pagos.loc[t,'años'] = vencimiento_años
                spot_fechas_pagos.loc[t,'tasa_spot'] = r_spot

            # ( 3 ) Encuentra rn
            r_iteracion = r(tasa_referencia=tasa_referencia_continua,cusips_mm=cus_mm,cusips_instrumentos=cusips_instrumentos,datos_instrumentos=datos_instrum)
            datos_bonos = datos_instrum.copy()
            datos_bonos.reset_index(inplace=True)
            datos_bonos.set_index('ticker',inplace=True)
            for bono in cusips_bonos:
                matriz = CF[bono]
                matriz.reset_index(inplace=True)
                matriz.set_index('payment_date',inplace=True)
                indice_matriz_sin_vencimiento = matriz.index
                indice_matriz_sin_vencimiento = list(indice_matriz_sin_vencimiento)
                indice_matriz_sin_vencimiento.pop(-1)
                for i in indice_matriz_sin_vencimiento:
                    #Siguientes iteraciones
                    matriz.loc[i,'factor_descuento'] = math.exp(-spot_fechas_pagos.loc[i,'tasa_spot']*spot_fechas_pagos.loc[i,'años'])
                matriz['DCF'] = matriz['Flujo_Total'] * matriz['factor_descuento']
                precio_bono_teorico = matriz['DCF'].sum()
                precio_teorico_sin_ultimo_pago = precio_bono_teorico - matriz.iloc[-1]['DCF']
                precio_bono_mkt = datos_bonos.loc[bono,'Precio_Sucio'] * FaceAmount / 100
                ultimo_pago = matriz.iloc[-1]['Flujo_Total']
                fecha_ultimo_pago = matriz.index[-1]
                def buscar_objetivo_DF(rn):
                    DF = ((precio_bono_mkt - precio_teorico_sin_ultimo_pago) / ultimo_pago) - rn
                    return DF
                df_n =  sp.broyden1(buscar_objetivo_DF,1)
                rn = math.log1p(1/df_n-1)
                años_ultimo_pago = ((fecha_ultimo_pago - st_dt).days) / 360
                rn = rn/ años_ultimo_pago
                spot_fechas_pagos.loc[fecha_ultimo_pago,'tasa_spot'] = rn
                spot_fechas_pagos.loc[fecha_ultimo_pago,'años'] = años_ultimo_pago
                posicion_bono = len(cusips_mm) + cusips_bonos.index(bono) + 2
                r_iteracion.loc[posicion_bono,'ticker'] = bono
                r_iteracion.loc[posicion_bono,'años'] = datos_bonos.loc[bono,'años']
                r_iteracion.loc[posicion_bono,'r'] = rn
                tasa_iteracion_anterior = tasas_nodos[bono]
                tasas_nodos[bono] = rn
                diferencias_tasas_nodos[bono] = rn - tasa_iteracion_anterior

            contador = contador + 1

    #Para asegurarse que el bootstrapping converja
    suma = 0
    for bono in diferencias_tasas_nodos.keys():
        suma += abs(diferencias_tasas_nodos[bono])

    if suma >= 0.00000001:
        print('No converge')



    #Creación carpeta
    
    path = path_dia(path_inicial=path_curva_spot,fecha=fecha)
    path_1 = path_final(path_con_fecha=path,economia=economia)
    path_completo = path_final_final(path_final=path_1,metodo=metodo)

    if os.path.isdir(path) == False:
        os.mkdir(path_curva_spot + fecha)
    else:
        ()
    
    if os.path.isdir(path_1) == False:
        os.mkdir(path_curva_spot + fecha + '/' + economia)
    else:
        ()

    if os.path.isdir(path_completo) == False:
        os.mkdir(path_curva_spot + fecha + '/' + economia + '/' + metodo)
    else:
        ()

    datos_instrum.to_csv(path_completo + 'datos_instrum_bootstrap.csv')
    r_iteracion.to_csv(path_completo + 'r_iteracion.csv')

    #Gráfica
    puntos_grafica = np.linspace(0.1,30,360)
    columnas = ['tasa_spot']
    indice = puntos_grafica
    graficas = pd.DataFrame(index=indice,columns=columnas)
    for punto in puntos_grafica:
        graficas.loc[punto] = tasa_spot(t=punto,rt=r_iteracion,f=f_iteracion,fd=fd_iteracion,metodo=metodo)*100
    graficas.to_excel(path_completo + 'grafica.xlsx')

    return