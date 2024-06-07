import sys 
#read argus output line by line 
import pandas as pd
import time 
import csv 
import os

def get_argus_output(filename): 
    trafic_df = pd.read_csv(filename, delimiter='[ ]+', engine='python')
    return trafic_df 


def get_ct_dst_sport_ltm(trafic_df): 
    """No of connections of the same destination
    address (3) and the source port (2) in 100
    connections according to the last time
    (26)"""
    trafic_df['ct_src_sport_ltm'] = None 
    for line in trafic_df.itertuples(index=True, name='Pandas'): 
        lastTime = line.LastTime
        matchinglines = trafic_df[(trafic_df['LastTime'] <= lastTime) & (trafic_df['Sport'] == line.Sport) & (trafic_df['DstAddr'] == line.DstAddr)].tail(100) 
        trafic_df.loc[line.Index, 'ct_src_sport_ltm']=matchinglines.shape[0]

    return  trafic_df

def get_ct_dst_src_ltm(trafic_df):
    """No of connections of the same source (1)
    and the destination (3) address in in 100
    connections according to the last time
    (26)."""
    trafic_df['ct_dst_src_ltm'] = None 
    for line in trafic_df.itertuples(index=True, name='Pandas'): 
        #get the 100 connections as compared to last time
        lastTime = line.LastTime
        matchinglines = trafic_df[(trafic_df['LastTime'] <= lastTime) & (trafic_df['SrcAddr'] == line.SrcAddr) & (trafic_df['DstAddr'] == line.DstAddr)].tail(100) 
        trafic_df.loc[line.Index, 'ct_dst_src_ltm']=matchinglines.shape[0]

    return  trafic_df

def get_ct_src_dport_ltm(trafic_df):
    """No of connections of the same source
    address (1) and the destination port (4) in
    100 connections according to the last time
    (26)."""
    trafic_df['ct_src_dport_ltm'] = None 
    for line in trafic_df.itertuples(index=True, name='Pandas'): 
        lastTime = line.LastTime
        matchinglines = trafic_df[(trafic_df['LastTime'] <= lastTime) & (trafic_df['SrcAddr'] == line.SrcAddr) & (trafic_df['Dport'] == line.Dport)].tail(100) 
        trafic_df.loc[line.Index, 'ct_src_dport_ltm']=matchinglines.shape[0]
    return  trafic_df

def get_ct_dst_ltm(trafic_df): 
    """No. of connections of the same
    destination address (3) in 100 connections
    according to the last time (26)."""
    trafic_df['ct_dst_ltm'] = None
    for line in trafic_df.itertuples(index=True, name='Pandas'):
        lastTime = line.LastTime
        matchinglines = trafic_df[(trafic_df['LastTime'] <= lastTime) & (trafic_df['DstAddr'] == line.DstAddr)].tail(100) 
        trafic_df.loc[line.Index, 'ct_dst_ltm']=matchinglines.shape[0]
    return trafic_df

def get_ct_state_ttl(trafic_df):
    """No. for each state (6) according to
    specific range of values for
    source/destination time to live (10) (11)."""
    trafic_df['ct_state_ttl'] = None
    for line in trafic_df.itertuples(index=True, name='Pandas'):
        if ((line.sTtl == 62 or line.sTtl == 63 or line.sTtl == 254 or line.sTtl == 255) and (line.dTtl == 252 or line.dTtl == 253) and line.State == 'FIN'  ):
            trafic_df.loc[line.Index, 'ct_state_ttl'] = '1'
        else:
            if ((line.sTtl ==0 or line.sTtl ==62 or line.sTtl==254) and (line.dTtl ==0) and line.State == 'INT'):
                trafic_df.loc[line.Index, 'ct_state_ttl'] = '2'
            else:
                if ((line.sTtl ==62 or line.sTtl==254) and (line.dTtl ==62 or line.dTtl == 252 or line.dTtl==253) and line.State == 'CON'):
                    trafic_df.loc[line.Index, 'ct_state_ttl'] = '3'
                else:
                    if ((line.sTtl ==254) and (line.dTtl ==252) and line.State == 'ACC'):
                        trafic_df.loc[line.Index, 'ct_state_ttl'] = '4'
                    else:
                        if ((line.sTtl ==254) and (line.dTtl ==252) and line.State == 'CLO'):
                            trafic_df.loc[line.Index, 'ct_state_ttl'] = '5'
                        else:
                            if ((line.sTtl ==254) and (line.dTtl ==0) and line.State == 'REQ'):
                                trafic_df.loc[line.Index, 'ct_state_ttl'] = '6'
                            else:
                                trafic_df.loc[line.Index, 'ct_state_ttl'] = '0'
    
    return trafic_df
                                   
def main():
    pcap_filename = 'datasets/unsw-nb15-pcap-files/2.pcap' # pcap file to be analyzed
    argus_ra_input_filename = 'output/ArgusFiles/argus_input.argus' #argus file produced from pcap file
    argus_ra_output_filename = 'output/ArgusFiles/argus_ra_output.csv' #argus output file produced from argus file
    #Commmand for producing argus file from pcap file
    start_time = time.time()
    produce_argus_file = ' argus -r ' + pcap_filename + ' -w ' + argus_ra_input_filename
    os.system(produce_argus_file)
    #command for executing argus-ra on argus file
    produce_argus_ra_output = 'ra -r ' + argus_ra_input_filename + ' -s dur, ltime, state,  saddr, daddr, sport, dport, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, smeansz, dmeansz'+ ' > ' + argus_ra_output_filename
    os.system(produce_argus_ra_output)
    output_filename = 'output/ArgusFiles/argus_output_features.csv'
    trafic_df = get_argus_output(argus_ra_output_filename)
    get_ct_dst_sport_ltm(trafic_df)
    get_ct_dst_src_ltm(trafic_df)
    get_ct_src_dport_ltm(trafic_df)
    get_ct_dst_ltm(trafic_df)
    get_ct_state_ttl(trafic_df)
    
    trafic_df = trafic_df.drop(['SrcAddr', 'DstAddr', 'Sport', 'Dport', 'State', 'LastTime'], axis=1)
   
    trafic_df.to_csv(output_filename, index=False, sep='\t')
    end_time = time.time()
    print('Time taken to extract features from pcap file: ', end_time - start_time)

if __name__ == '__main__' :
    main()
