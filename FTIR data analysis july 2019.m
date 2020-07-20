clc
clear all, close all
addpath('I:\presentation_golam\FTIR')
folder = 'I:\presentation_golam\FTIR'
names = dir(folder);
end_data(500,1) = struct();
counter = 0;
for a=5:4:33
     counter = counter+1
    sample = names(a).name;
    [wn, data] = ftir_load(sample);  
 I = data(:,:,50);
%I = histeq(I);
level_prefill =multithresh(I);
seg_I_single_prefill = imquantize(I,level_prefill);
I_thresholded = seg_I_single_prefill - 1;

% fill holes in the image
I_fill = imfill(I_thresholded,'holes');
erased = bwconncomp(I_fill);
[max_size, max_index] = max(cellfun('size', erased.PixelIdxList, 1));
largest_sample = erased.PixelIdxList(max_index);
largest_sample{1, 1};
new_image = zeros(size(I,1),size(I,2));
new_image(erased.PixelIdxList{max_index}) = 1; 
    
   
masked_image_1 = (new_image).*data(:,:,:);

% imagesc(masked_image_1(:,:,66));
    if a==5 
    J = imrotate(masked_image_1,175);
    
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[109 49 50 23])
   L=109:159
   y=49:72
   figure;
   imagesc(J(y,L,66));
   title('region center')
   
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_5=squeeze(mean(data_row_center,2));
    end_data(counter).data_row_center_mean_a_5 = data_row_center_mean_a_5;
   
    else if a==9
    J = imrotate(masked_image_1,180);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[22 6 50 20])
    L=22:72
    y=6:26  
    figure;
   imagesc(J(y,L,66));
   title('region center')
        
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_9=squeeze(mean(data_row_center,2));
     
        else if  a==13
     J = imrotate(masked_image_1,356);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[80 32 50 38])
       L=80:130
       y=32:70
   figure;
   imagesc(J(y,L,66));
   title('region center k')
            
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_13=squeeze(mean(data_row_center,2));
  end_data(counter).data_row_center_mean_a_13 = data_row_center_mean_a_13;
        else if  a==17
    J = imrotate(masked_image_1,178);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[87 20 50 20])
     L=87:137
     y=20:40
   figure;
   imagesc(J(y,L,66));
   title('region center ')
   
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_17=squeeze(mean(data_row_center,2));
    end_data(counter).data_row_center_mean_a_17 = data_row_center_mean_a_17;
            else if  a==21
    J = imrotate(masked_image_1,175);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[82 54 50 28])
     L=82:132
     y=54:82
   figure;
   imagesc(J(y,L,66));
   title('region center ')
   
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_21=squeeze(mean(data_row_center,2));
    
           else if  a==25
    J = imrotate(masked_image_1,15);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[103 79 50 50])
    L=103:153
    y=79:129
   figure;
   imagesc(J(y,L,66));
   title('region center ')
   
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_25=squeeze(mean(data_row_center,2));
    end_data(counter).data_row_center_mean_25 = data_row_center_mean_25;
              else if  a==29
    J = imrotate(masked_image_1,171);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[108 31 50 25])
    L=108:158
    y=31:56
   figure;
   imagesc(J(y,L,66));
   title('region center ')
   
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_29=squeeze(mean(data_row_center,2));
    
     else  a==33
    J = imrotate(masked_image_1,10);
    figure;
    imagesc(J(:,:,66));
    extract_region=imrect(gca,[90 60 50 40])
    L=90:140
    y=60:100
   figure;
   imagesc(J(y,L,66));
   title('region center ')
   
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a_33=squeeze(mean(data_row_center,2));
    end_data(counter).data_row_center_mean_a_33 = data_row_center_mean_a_33;
                  end
               end
               end
               end
       end
       end
       end
     
      

                

%% calculating average Amide I absorption for regions from surface to bottom
    
    region_center=(J(y,L,:));
    data_row_center=squeeze(region_center);
    data_row_center_mean_a=squeeze(mean(data_row_center,2));
    figure;
    permute_data=permute(J,[2 1 3]);
    subplot(211)
    imagesc(permute_data(L,y,245));
     axis image
    title('amide I')
    subplot(212)
    plot(data_row_center_mean_a(:,245));
     axis([0 28 0 1.4])
    
% calculating average Amide II absorption for regions from surface to bottom
    figure;
    permute_data=permute(J,[2 1 3]);
    subplot(211)
    imagesc(permute_data(L,y,218));
    axis image
    title('amide II')
    subplot(212)
    plot(data_row_center_mean_a(:,218));
    axis([0 28 0 1])
% calculating average Amide III absorption for regions from surface to bottom

    figure;
    permute_data=permute(J,[2 1 3]);
    subplot(211)
    imagesc(permute_data(L,y,137));
     axis image
    title('amide III')
    subplot(212)
    plot(data_row_center_mean_a(:,137));    
     axis([0 28 0 1])
% calculating depth-wise average proteoglycan peak(1064cm-1) absorption for regions from surface to bottom

    figure;
    subplot(211)
    imagesc(permute_data(L,y,92));
     axis image
    title('proteoglycan peak');
    subplot(212)
    plot(data_row_center_mean_a(:,92));

     axis([0 28 0 1])
%% Calculate depth-wise average collagen integrity ratio  

Range_3_center = mean(J(y,L,106:121),3);%range for amide peak I in between(1582 - 1715)
Range_4_center = mean(J(y,L,153:179),3);%range for amide peak II in between(1485 -1585)

permuted_collagen_int_center_5=permute((Range_3_center./Range_4_center),[2 1]); %collagen integrity ratio for region center
     figure;   
  
    subplot(211)
    imagesc(permuted_collagen_int_center_5);
   axis image
   
    title('collagen integrity ratio')
    subplot(212)
  
    plot(mean(permuted_collagen_int_center_5));
   axis([0 28 0.4 1])
   
%% clustering   
 c=J(y,L,:);
      figure;
     imagesc(mean (c,3));
% set(gca,'ydir','normal');
     title(' Before Cluster')
   axis image;
   %truncate spectrum
    c = c(:,:,49:283);
    wn = wn(49:283);
    % vector normalization 
    re_data = vectornorm_map(c);
      
    
    size_data = size(c);
    re_data = reshape(c,size_data(1)*size_data(2),size_data(3));
    
    % truncate spectra and wavenumber vector
     
    %re_data = re_data(:,49:283);
   % wn_truncated = wn(49:283);
    
   
    % vector normalization 
    %re_data = vectornorm_map(re_data);
    
    % K-means clustering
    idx = kmeans(re_data,5);
    CI = reshape(idx,[size_data(1) size_data(2)]);
    figure;
    imagesc(CI);
   %   set(gca,'ydir','normal');
    axis image;
    title(' After Cluster')
    h = colorbar;
    h.Ticks = 1:5;
    CS = zeros(5,size_data(3));
    figure;
    
        CS(1,:) = mean(re_data (idx == 1,:),1);
        plot(wn,CS(1,:),'b');
        hold on;
    CS(2,:) = mean(re_data (idx == 2,:),1);
        plot(wn,CS(2,:),'g');
        
        hold on
          CS(3,:) = mean(re_data (idx == 3,:),1);
        plot(wn,CS(3,:),'r');
        
         hold on
          CS(4,:) = mean(re_data (idx == 4,:),1);
        plot(wn,CS(4,:),'c');
        
        
         hold on
          CS(5,:) = mean(re_data (idx == 5,:),1);
        plot(wn,CS(5,:),'m');
        
        
    hold off;
    title('Mean spectra of clusters');
         ylabel(' Raw absorbance');
    xlabel('Wavenumber ( per cm)')
    axis tight;  
    
    
% CS = zeros(8,size_data(3));
% figure
% for a=5:4:33
%     n=21
%     x=linspace(0,30*pi,n)
%     region_center=(J(y,L,:));
%     data_row_center=squeeze(region_center);
%     data_row_center_mean_a=squeeze(mean(data_row_center,2));
%      CS(a,:) =data_row_center_mean_a (:,245);
%      randomperutation=randperm(n)
%      percentkept=100
%      numkept=round(n*percentkept/100)
%      
% xkept= x(randomperutation(1:numkept))
% ykept= y(randomperutation(1:numkept))
% yinterp=interp1(xkept,ykept,x,'pchip')
% figure;
% plot(x,yinterp,'r',xkept,ykept,'ro')
% hold on
 end

%% amide 1 
%control sample
n=21
x=linspace(0,30*pi,n)
y1 = data_row_center_mean_a_5(:,245); 
y2 = data_row_center_mean_a_13(:,245);
y3 = data_row_center_mean_a_17(:,245);
y4 = data_row_center_mean_25(:,245);
y5 = data_row_center_mean_a_33(:,245);


randomperutation=randperm(n)
percentkept=100
numkept=round(n*percentkept/100)

xkept= x(randomperutation(1:numkept))
y1kept= y1(randomperutation(1:numkept))
y2kept= y2(randomperutation(1:numkept))
y3kept= y3(randomperutation(1:numkept))
y4kept= y4(randomperutation(1:numkept))
y5kept= y5(randomperutation(1:numkept))

y1interp=interp1(xkept,y1kept,x,'pchip')
y2interp=interp1(xkept,y2kept,x,'pchip')
y3interp=interp1(xkept,y3kept,x,'pchip')
y4interp=interp1(xkept,y4kept,x,'pchip')
y5interp=interp1(xkept,y5kept,x,'pchip')

figure;
plot(x,y1interp,'r',xkept,y1kept,'ro')
hold on
plot(x,y2interp,'g',xkept,y2kept,'ro')
hold on
plot(x,y3interp,'b',xkept,y3kept,'ro')
hold on
plot(x,y4interp,'m',xkept,y4kept,'ro')
hold on
plot(x,y5interp,'k',xkept,y5kept,'ro')


%% standard deviation for controol sample

n=21
x=linspace(0,100,n)
zk=mean([y1interp;y2interp;y3interp;y4interp;y5interp]);
z=([y1interp;y2interp;y3interp;y4interp;y5interp])
y=std(z)
figure;
errorbar(x,zk,y,'-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')


%% damage sample
n=21
x=linspace(0,30*pi,n)
y1 = data_row_center_mean_a_9(:,245); 
y2 = data_row_center_mean_a_21(:,245);
y3 = data_row_center_mean_a_29(:,245);


randomperutation=randperm(n)
percentkept=100
numkept=round(n*percentkept/100)

xkept= x(randomperutation(1:numkept));
y1kept= y1(randomperutation(1:numkept))
y2kept= y2(randomperutation(1:numkept))
y3kept= y3(randomperutation(1:numkept))


y1interp=interp1(xkept,y1kept,x,'pchip')
y2interp=interp1(xkept,y2kept,x,'pchip')
y3interp=interp1(xkept,y3kept,x,'pchip')

figure;
plot(x,y1interp,'r',xkept,y1kept,'ro')
hold on
plot(x,y2interp,'g',xkept,y2kept,'ro')
hold on
plot(x,y3interp,'b',xkept,y3kept,'ro')

%% standard deviation for damage

n=21
x=linspace(0,100,n)
yk=mean([y1interp;y2interp;y3interp])
z=([y1interp;y2interp;y3interp])
y=std(z)

err = .06*ones(size(y));
figure;
errorbar(x,yk,y,'-s','MarkerSize',5,...
    'MarkerEdgeColor','red','MarkerFaceColor','red')
 


