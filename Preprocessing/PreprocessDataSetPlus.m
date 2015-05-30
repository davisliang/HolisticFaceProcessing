function [] = PreprocessDataSetPlus(topFocus, fullFocus, loc, imshow, num_each, num_test, matPATH)

%   PreprocessDataSet()
%   Training and test set assembly for training TM ("The Model", Dailey and Cottrell (1999))
%   Author: Panqu Wang
%   This is only a toy version. Do not distribute without permission.

%1: AFRAID
%2: ANGRY
%3: DISGUSTED
%4: HAPPY
%5: NEUTRAL
%6: SAD
%7: SUPRISED

% Finding location of data set.
cd(loc);
objectparent=dir(pwd);
preprocessedData=[];

for objectname=3:length(objectparent)
    name=objectparent(objectname).name;
    cd(name)
    object=dir(pwd);

    test_list=randperm(num_each,num_test);
    total_num=(length(object)-2)*num_each;
    total_train=(length(object)-2)*(num_each-num_test);
    total_test=(length(object)-2)*(num_test);
    num_pca=8;
    size_gb=48;
    std_dev=pi;
    dwsp=8;%downsampled rate
    
    display 'Start Preprocessing...'

    %% for each subject (Tom, Mary, ...) in given object (faces, ...)
    for i=3:length(object)
        display(['Subject ' num2str(i-2)]);
        cd(object(i).name);
        subject=dir(pwd);
        nimage(i-2)=length(subject)-2;%number of training image
        order{i-2}=1:nimage(i-2);
        %% for each image in given subject, do gabors_filtering
        trainIndex=1;
        testIndex=1;
        for current_number=1:num_each
            f=imread(subject(order{i-2}(current_number)+2).name);
            
            %imageShow(f); %DAVIS DEBUGGING STUFF***********************************************
            
            if size(size(f),2)==2
                f=imresize(im2double(f),[64 64]);
            else
                f=rgb2gray(im2double(f)); 
                f=imresize(f,[64 64]);
            end
            [height width]=size(f); 
            %scale
            for temp=1:5
                k(temp)=(2*pi/width)*2^temp;
            end
            %orientations
            for temp=1:8
                phi(temp)=(pi/8)*(temp-1);
            end
            %constructing gabors filter (16*16) and filtering the input image
            %figure;
            for scale=1:size(k,2)
            %scale
                for orientations=1:size(phi,2)
                    for ii=-size_gb+1:size_gb
                        for j=-size_gb+1:size_gb
                            carrier(ii+size_gb,j+size_gb)=exp(1i*(k(scale)*cos(phi(orientations))*ii+k(scale)*sin(phi(orientations))*j));
                            envelop(ii+size_gb,j+size_gb)=exp(-(k(scale)^2*(ii^2+j^2))/(2*std_dev*std_dev));
                            gabors(ii+size_gb,j+size_gb,orientations)=carrier(ii+size_gb,j+size_gb)*envelop(ii+size_gb,j+size_gb);
                        end
                    end
                     %subplot(2,4,orientations);
                     %imshow(gabors(:,:,orientations), []);
                    f_filtered{scale}(:,:,orientations)=imfilter(f,gabors(:,:,orientations),'replicate','conv');       
%                     imshow(f_filtered{scale}(:,:,orientations),[])
%%
                    f_filtered{scale}(:,:,orientations);
                    
%                     imshow(f_filtered{scale}(:,:,orientations),[])



                end
                
                %now we have 8 orientationss for each scale, downsample and do normalization
                for orientations=1:size(phi,2)
                   f_filtered_dwsp{scale}(:,:,orientations)=imresize(f_filtered{scale}(:,:,orientations),[dwsp dwsp]);
                end
                for orientations=1:size(phi,2)
                   f_filtered_normalized_dwsp{scale}(:,:,orientations)=abs(f_filtered_dwsp{scale}(:,:,orientations))./sum(abs(f_filtered_dwsp{scale}),3);               

                end
            end
            for scale=1:size(k,2);
                    f_filtered_normalized_dwsp_vector((scale-1)*dwsp*dwsp*size(phi,2)+1:(scale)*dwsp*dwsp*size(phi,2),current_number)=f_filtered_normalized_dwsp{scale}(:);             
            end      
            
            %testset and trainingset assembly
            if ismember(current_number,test_list)
                f_filtered_normalized_dwsp_vector_allsub_test(:,(i-3)*num_test+testIndex)=f_filtered_normalized_dwsp_vector(:,current_number);
                testIndex=testIndex+1;
            else
                f_filtered_normalized_dwsp_vector_allsub_train(:,(i-3)*(num_each-num_test)+trainIndex)=f_filtered_normalized_dwsp_vector(:,current_number);
                trainIndex=trainIndex+1;
            end
        end
        clear f_filtered_normalized_dwsp_vector;
        cd ..
    end

    %PCA on different scale
    for scales=1:size(k,2) 
        scale_all(:,:,scales)=f_filtered_normalized_dwsp_vector_allsub_train((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
        scale_all_test(:,:,scales)=f_filtered_normalized_dwsp_vector_allsub_test((scales-1)*dwsp*dwsp*size(phi,2)+1:scales*dwsp*dwsp*size(phi,2),:);
        mean_images(:,scales)=mean(scale_all(:,:,scales),2);    
        
        %turk and pentland trick, for each scale
        mean_subst(:,:,scales)=scale_all(:,:,scales)-repmat(mean_images(:,scales),1,total_train);
        mean_subst_test(:,:,scales)=scale_all_test(:,:,scales)-repmat(mean_images(:,scales),1,total_test);
        cov_scale=(mean_subst(:,:,scales)'*mean_subst(:,:,scales))*(1/total_train); %(estimate of covariance)
        [vector_temp value]=eig(cov_scale);
        vector_biggest=vector_temp(:,end-num_pca+1:end);
        
        %original principal components
        vector_ori(:,:,scales)=mean_subst(:,:,scales)*vector_biggest;

        %projection onto the basis vector vector_ori(dimension 512-dimension 8)
        %normal
        f_PCA_scale_normal=zscore(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
        f_PCA_scale_test_normal=zscore(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));
        
%         f_PCA_scale_normal=(vector_ori(:,:,scales)'*mean_subst(:,:,scales));
%         f_PCA_scale_test_normal=(vector_ori(:,:,scales)'*mean_subst_test(:,:,scales));        
        
        f_PCA_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_normal;
        f_PCA_test_temp_normal((scales-1)*num_pca+1:scales*num_pca,:)=f_PCA_scale_test_normal;
      
    end
    
    for i=1:length(object)-2
        f_PCA_normal_DATASET_train{i}=f_PCA_temp_normal(:,(i-1)*(num_each-num_test)+1:i*(num_each-num_test));
        f_PCA_normal_DATASET_test{i}=f_PCA_test_temp_normal(:,(i-1)*num_test+1:i*num_test);  
    end
    
    preprocessedData(objectname-2).name=name;
    preprocessedData(objectname-2).trainingSet=f_PCA_normal_DATASET_train;
    preprocessedData(objectname-2).testSet=f_PCA_normal_DATASET_test;
    cd ..
end
    display 'finished.'
    save([strcat('../',matPATH)],'preprocessedData')
    cd ..
end