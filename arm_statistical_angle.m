
num_ant=256; % total number of antenna

beam_ant_num=[4 8 16 32 64 128 192 256]; % number of antenna for different beamwidths
num_beamwidth=length(beam_ant_num);

num_chan=1000;
K=100;
AOA=2*pi/3*(rand(num_chan,1)-0.5); % angle of arrival (AoA)
channel=(sqrt(K)*exp(-1i*pi*sin(AOA)*[0:(num_ant-1)]).'+(randn(num_ant,num_chan)+1i*randn(num_ant,num_chan))/sqrt(2))/sqrt(num_ant);%channel vector
channel=channel*diag(1./sqrt((diag(channel'*channel))));

sigma2=0.00025; % standard deviation of noise
sigma2_chan=0.02; % standard deviation of AoA error

num_est=1000; % number of realizations
snr_func={[1 0.2 0.004],
[1 1 0.3 0.02 0.001],
[1 1 1 1 0.2 0.02 0.0025], 
[1 1 1 1 1 1 0.8 0.07 0.004],
[1 1 1 1 1 1 1 1 0.25 0.018 0.0035],
[1 1 1 1 1 1 1 1 1 1 0.375 0.047 0.004 0.0015],
[1 1 1 1 1 1 1 1 1 1 0.9 0.28 0.045 0.01 0.0045 0.001],
[1 1 1 1 1 1 1 1 1 1 1 1 1 0.8 0.15 0.02 0.004 0.002],
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.95 0.3 0.06 0.015 0.0065 0.0015],
[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0.45 0.14 0.03 0.017 0.0075 0.004 0.002]};
SNR=zeros(num_chan,num_est);
for i_beam=1:num_beamwidth
    for i=1:num_est
        steer_vec_n=exp(1i*pi*sin(AOA+sigma2_chan*(randn(num_chan,1)))*[0:(num_ant-1)]); %steering vector with pointing error
        steer_vec_n(:,beam_ant_num(i_beam)+1:end)=0;
        power_scale=1./(diag(steer_vec_n*steer_vec_n'));

        signal_com=abs(diag(steer_vec_n*channel)).^2.*power_scale;
        SNR(:,i)=10*log10(signal_com./(sigma2));
    end

    for i=1:length(snr_func)
        snr_fit=fit([1:length(snr_func{i})]',10*log10([snr_func{i}])','linearinterp');
        PER=snr_fit(SNR);
        PER(PER>0)=0;
        succ_indicator=1-10.^(PER./10);
        succ_prob(i,i_beam)=mean(mean(succ_indicator));
    end
end
rate_vec=[1386 1732.5 2079 2772 3465 4158 4504.5 5197.5 6237 6756.75]';
lambda_d=diag(rate_vec)*succ_prob;
[va,idx]=max(lambda_d(:));
[rate_idx,beam_idx]=ind2sub(size(lambda_d),idx);
max_throu_succ=succ_prob(rate_idx,beam_idx);
save('prob_data_48163264128192256_normal0.02_0.00025.mat')
