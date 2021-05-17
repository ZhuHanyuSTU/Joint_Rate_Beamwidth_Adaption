%% setting parameters
epsilon=5; delta=0.01;    %threshold for constraint

succ_prob=load('prob_data_48163264128192256_normal0.05_0.00025.mat','succ_prob');


succ_prob=succ_prob.succ_prob;
succ_prob=succ_prob(:,end);
rate_vec=[1386 1732.5 2079 2772 3465 4158 4504.5 5197.5 6237 6756.75]';

K=length(rate_vec);   %number of rates
N=1;   %number of beamwidths
D=N*K; %num of arms

num_fram_pt=[1 1 1 1 1 1 1 1 1 1]';%number of transmitted frame in one time slot 

%% statistical

lambda_d=diag(rate_vec)*succ_prob;    %expectation of arms reward
lambda_d=lambda_d(:);
Lambda_d=diag(rate_vec.^2)*succ_prob.*(1-succ_prob)./num_fram_pt;    %variance of arms reward
Lambda_d=Lambda_d(:);

num_fram_pt_v=kron(ones(D,1),num_fram_pt);
rate_vec_v=kron(ones(D,1),rate_vec);
succ_prob_v=succ_prob(:);

%% algorithm

T=20e4;                %number of time slot
R_d=zeros(D,T);         %R_d(d,t) is the reward for arm d in time slot t
T_d=zeros(D,1);         %T_d(d) is the number of time that d is pulled
a_t=zeros(T,1);      %a_t(t) is the arm pulled in time slot t
attem_d=zeros(D,1);
succ_d=zeros(D,1);

figure(1)
for t=1:D
    a_t(t)=t;
    [succ_indicator]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
    attem_d(t)=attem_d(t)+1;
    succ_d(t)=succ_d(t)+succ_indicator;
    
    
    R_d(a_t(t),t)=succ_indicator*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
    R(t)=R_d(a_t(t),t);
    T_d(a_t(t))=T_d(a_t(t))+1;  
    con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
end
probe_rate=0.1;
n_total=0;
n_probe=0;
n_deferred=0;
update_slot=t;
update_interval=0;
weight_update=1;
succ_prob_est=succ_d./attem_d;
tp=succ_prob_est.*rate_vec;

num_retry=2;
while( t<=T)
    [sorted_tp,sorted_tp_arm]=sort(tp,'descend');
    [sorted_p,sorted_p_arm]=sort(succ_prob_est,'descend');
    maxtp=sorted_tp_arm(1);
    tp2=sorted_tp_arm(2);
    lowest=sorted_tp_arm(end);
    maxp=sorted_p_arm(1);
    

    count1=0;
    count2=0;
    count3=0;
    count4=0;
    succ_indicator=0;
    n_total=t-1;
    if n_total*probe_rate-n_probe+n_deferred/2>0
        probe=ceil(D*rand(1));
        n_probe=n_probe+1;
        if tp(probe)>tp(maxtp)
            chain1=probe;
            chain2=maxtp;
        else
            chain1=maxtp;
            chain2=probe;
        end
    else
        chain1=maxtp;
        chain2=tp2;
    end
    while(((succ_indicator==1)||(count1>num_retry))==0)
        a_t(t)=chain1;
        [succ_indicator]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
        attem_d(a_t(t))=attem_d(a_t(t))+1;
        succ_d(a_t(t))=succ_d(a_t(t))+succ_indicator;
        R_d(a_t(t),t)=succ_indicator*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
        R(t)=R_d(a_t(t),t);
        T_d(a_t(t))=T_d(a_t(t))+1;  
        con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
        ave_reward(t)=mean(R(1:t));
        t=t+1;
        count1=count1+1;
    end
    if succ_indicator==0
        while(((succ_indicator==1)||(count2>num_retry))==0)
            a_t(t)=chain2;
            [succ_indicator]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
            attem_d(a_t(t))=attem_d(a_t(t))+1;
            succ_d(a_t(t))=succ_d(a_t(t))+succ_indicator;
            R_d(a_t(t),t)=succ_indicator*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
            R(t)=R_d(a_t(t),t);
            T_d(a_t(t))=T_d(a_t(t))+1;  
            con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
            ave_reward(t)=mean(R(1:t));
            t=t+1;
            count2=count2+1;
        end  
    end
    if succ_indicator==0
        while(((succ_indicator==1)||(count3>num_retry))==0)
            a_t(t)=maxp;
            [succ_indicator]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
            attem_d(a_t(t))=attem_d(a_t(t))+1;
            succ_d(a_t(t))=succ_d(a_t(t))+succ_indicator;
            R_d(a_t(t),t)=succ_indicator*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
            R(t)=R_d(a_t(t),t);
            T_d(a_t(t))=T_d(a_t(t))+1;  
            con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
            ave_reward(t)=mean(R(1:t));
            t=t+1;
            count3=count3+1;
        end  
    end
    if succ_indicator==0
        while(((succ_indicator==1)||(count4>num_retry))==0)
            a_t(t)=lowest;
            [succ_indicator]=arm_reward(succ_prob_v(a_t(t)),num_fram_pt_v(a_t(t)));
            attem_d(a_t(t))=attem_d(a_t(t))+1;
            succ_d(a_t(t))=succ_d(a_t(t))+succ_indicator;
            R_d(a_t(t),t)=succ_indicator*rate_vec_v(a_t(t))/num_fram_pt_v(a_t(t));
            R(t)=R_d(a_t(t),t);
            T_d(a_t(t))=T_d(a_t(t))+1;  
            con_ind(t)=sum(R(1:t)<epsilon)-t*delta;
            ave_reward(t)=mean(R(1:t));
            t=t+1;
            count4=count4+1;
        end  
    end
    if t-update_slot>=update_interval
        
        for i_update=1:D
            if attem_d(i_update)>0
                succ_prob_est(i_update)=succ_prob_est(i_update)*(1-weight_update)+weight_update*succ_d(i_update)/attem_d(i_update);
            end
        end
        tp=succ_prob_est.*rate_vec;
    end
    
end