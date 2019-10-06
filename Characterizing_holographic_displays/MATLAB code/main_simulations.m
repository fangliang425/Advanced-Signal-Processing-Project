% SGN-26006 Advanced signal processing laboratory (2019)
% 
% Assignment: Characterizing Holographic Displays
% Instructor: Jani Mäkinen
%
% Main function

clearvars

%% Parameters

lambda  = 534e-9; % green wavelength
eye_loc = [0, 2e-2]; % location of the simulated eye [x,z], by default set for near-eye applications
D_eye   = 5e-3; % pupil size (default 5 mm)

% Note the definition of z (negative z_f -> eye focused behind the hologram)
z_f = -400e-3; % z coordinate where the eye is focused

methods_all = {'Fresnel', 'RS', 'HS'}; % Hologram synthesis methods
method = methods_all{1}; % choose method (1 to 3)

% Hologram plane parameters
X_x = 2e-6; % pixel size (2µm)
N   = 2^13; % number of pixels
W_x = N*X_x; % physical size (of hologram)

%% Hologram plane

% Hologram plane (decentered) grid
x_max = X_x*N/2;
x = -x_max:X_x:x_max-X_x; % location of each hologram sample (x)
z_c = 0; % z location of the hologram

% HS parameters
seg_size_px = 64; % segment size in pixels
seg_size = seg_size_px*X_x; % segment physical size
x_c_ar = X_x*seg_size_px*(-(N/seg_size_px-1)/2:(N/seg_size_px-1)/2); % grid of the segment centers
num_seg = length(x_c_ar);

%% Input object (the point source)

% Note the definition of z (negative z_p -> point behind the hologram)
z_p = -400e-3; % point position in z (400 mm behind the hologram)
x_p = 0;
a_p = 1; % point amplitude
phi_p = 0; % relative phase

%% Check hologram size and sampling

if (0-z_p)/(eye_loc(2)-z_p)*D_eye > W_x
    warning('Hologram size is too small')
end

if lambda*eye_loc(2)/X_x/2 < abs(eye_loc(1))+D_eye/2
    warning('Hologram sampling is too coarse')
end

%% Synthesize the hologram

% Initialize the fields with zeros
obj_in = zeros(1,N);
% Spherical illumination
spherical_illum = exp(1j*2*pi/lambda*sqrt((x-eye_loc(1)).^2+(z_c-eye_loc(2)).^2));
spherical_illum_comp = exp(-1j*2*pi/lambda*sqrt((x-eye_loc(1)).^2+(z_c-eye_loc(2)).^2));

if z_p==0
    % Special case, point at hologram plane
    obj_in(N/2)=a_p*exp(1i*phi_p);
else
    % Choose the hologram synthesis method
    k=2*pi/lambda;
    switch method
        case 'Fresnel'
            % ADD YOUR CODE HERE
            inside=1j*k*((x-x_p).^2)/(2*(z_c-z_p));
            obj_in=exp(1j*k*(z_c-z_p))/sqrt(1j*lambda*(z_c-z_p))*exp(inside);
            obj_in = obj_in.*spherical_illum_comp; % for the spherical illumination, do not remove!
        case 'RS'
            % ADD YOUR CODE HERE
            w = (z_c-z_p)./(sqrt(1j*lambda)*((x-x_p).^2+(z_c-z_p)^2).^(3/4));
            obj_in=w.*exp(2*pi*1j/lambda*sqrt((x-x_p).^2+(z_c-z_p)^2));
            obj_in = obj_in.*spherical_illum_comp; % for the spherical illumination, do not remove!
        case 'HS'
            for ii=1:num_seg
                % hogel = holographic element
                x_c = x_c_ar(ii); % hogel center coordinate
                
                seg_ind_x = round(x_c/X_x+N/2)-seg_size_px/2+1:round(x_c/X_x+N/2)+seg_size_px/2;
                seg_x = x(seg_ind_x); % pixels x coordinates of the hogel
                
                ang_p_x_c = atand((x_c-x_p)./(z_c-z_p)); % angle between the point and hogel center
                f_x = sind(ang_p_x_c)/lambda; % spatial frequency
                
                r_p = sqrt((x_c-x_p).^2+(z_c-z_p)^2); % Euclidean distance between the point source and the hogel center
                spherical_illum_seg = exp(-1j*2*pi/lambda*sqrt((seg_x-...
                    eye_loc(1)).^2+(z_c-eye_loc(2)).^2));
                
                % calculate field inside the current hogel
                obj_in(seg_ind_x) = obj_in(seg_ind_x) + ...
                    a_p/r_p*exp(1i*2*pi*(f_x*(seg_x-x_c))+phi_p)...
                    .*spherical_illum_seg;
            end
    end
end

%% Run the simulation to obtain the retinal image

% Spherical illumination
obj_in = obj_in.*spherical_illum;  % the object field just after the hologram, when spherical illumination is used

% 1D field propagation
[PSF, sensor, X_u] = propagateField_PWD(obj_in, lambda, X_x, N, eye_loc, z_f, D_eye);

% Plot the PSF samples
%ADD YOUR CODE HERE
figure
plot(sensor,PSF);
title(' PSF Samples Plot when z_f=-400e-3')
xlabel('Sensor Position')
ylabel('PSF')

%% Analysis of results

MTF = abs(fftshift(fft(PSF))) / sum(PSF); % (normalized) MTF as a function spatial frequency

% Spatial frequency at which the MTF is to be evaluated
mtf_freq_c = 15; % cpd
mtf_freq_range = mtf_freq_c + [-0.5 0.5]*mtf_freq_c/4; % cpd (a small range of frequencies)

ind = round(length(MTF)/2+ length(MTF)/2 * 2*X_u./(2*tand(0.5)*25e-3./mtf_freq_range)); % indices corresponding to the frequency range
mtf_at_freq_c = mean(MTF(ind(1):ind(2))); % integrated over a small range around the exact frequency

% TASK:
% The variable mtf_at_freq_c contains a single value. Repeat the simulations 
% for different z_f values, fit a function to the data points and plot the 
% results (see more details in the instructions). You can change the code
% to do the repetition for you.

%Answer 2
% Repeat the process for varying values of zf (In this place, we set 7 points).
% Plot the PSF in one figures and print the MTF values.
figure
mtf_at_freq_c_s = zeros(7,1);
PSF_s = zeros(7,1535);
sensor_s = zeros(7,1535);
z_f_s=[-700e-3:100e-3:-100e-3];
hold on
for jj = [1:7]
    z_f=z_f_s(jj);
    [PSF, sensor, X_u] = propagateField_PWD(obj_in, lambda, X_x, N, eye_loc, z_f, D_eye);
    PSF_s(jj,:)=PSF;
    sensor_s(jj,:)=sensor;
    txt = ['z_f = ',num2str(z_f)];
    plot(sensor,PSF,'DisplayName',txt);
    title(' PSF Samples Plot')
    xlabel('Sensor Position')
    ylabel('PSF')
    
    MTF = abs(fftshift(fft(PSF))) / sum(PSF); 
    mtf_freq_c = 15; 
    mtf_freq_range = mtf_freq_c + [-0.5 0.5]*mtf_freq_c/4; 

    ind = round(length(MTF)/2+ length(MTF)/2 * 2*X_u./(2*tand(0.5)*25e-3./mtf_freq_range)); 
    mtf_at_freq_c = mean(MTF(ind(1):ind(2))); 
    mtf_at_freq_c_s(jj)=mtf_at_freq_c;
    fprintf('mtf_at_freq_c %d\n', mtf_at_freq_c)
end
hold off
legend show

%To make plots more clear
figure
for jj = [1:7]
    subplot(4,2,jj);
    plot(sensor_s(jj,:),PSF_s(jj,:));
    title([' PSF Samples Plot when z_f=',num2str( z_f_s(jj))])
    xlabel('Sensor Position')
    ylabel('PSF')
end
 %%
 %Answer 3
 % The following part is responsible for calculating the MTF values around certain spatial frequencies for each value of zf . 
 % And plot these values as a function of zf 
figure
z_f_s=[-700e-3:50e-3:-100e-3];
mtf_at_freq_c_s = zeros(13,1);
for jj = [1:13]
    z_f=z_f_s(jj);
    [PSF, sensor, X_u] = propagateField_PWD(obj_in, lambda, X_x, N, eye_loc, z_f, D_eye);
    MTF = abs(fftshift(fft(PSF))) / sum(PSF); 
    mtf_freq_c = 15; 
    mtf_freq_range = mtf_freq_c + [-0.5 0.5]*mtf_freq_c/4; 

    ind = round(length(MTF)/2+ length(MTF)/2 * 2*X_u./(2*tand(0.5)*25e-3./mtf_freq_range)); 
    mtf_at_freq_c = mean(MTF(ind(1):ind(2))); 
    mtf_at_freq_c_s(jj)=mtf_at_freq_c;
end

z_f_s_m=reshape(z_f_s,[13,1]);
mtf_at_freq_c_s_m=reshape(mtf_at_freq_c_s,[13,1]);
f1= fit(z_f_s_m,mtf_at_freq_c_s_m,'fourier5');
plot(f1,z_f_s_m,mtf_at_freq_c_s_m);
hold on
z_f_range=[-700e-3:1e-3:-100e-3];
y = feval(f1,z_f_range);
index = find(y==max(y));
plot(z_f_range(index),max(y),'r*','DisplayName','Maximum')
labelstr = sprintf('(%.2f,%.2f)', z_f_range(index),max(y));
text(z_f_range(index),max(y),labelstr,'VerticalAlignment','top','HorizontalAlignment','left')
title(' MTF at certain z_f')
xlabel('z_f')
ylabel('MTF')
grid on

