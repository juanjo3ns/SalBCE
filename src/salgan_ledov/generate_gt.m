clear all;
tic

load('VideoNameList.mat','VideoNameList')
videonumindex = 1;
% InputVideoName='animal_elephant01.mp4';
% InputVideoName_short=InputVideoName(1:end-4);

 % for m=1:length(VideoNameList)
     % if strcmp(InputVideoName_short,VideoNameList{m})==1
         % index=m;
         % break;
     % end
 % end
 for m = 1:length(VideoNameList)
	InputVideoName_short=VideoNameList{m};
	InputVideoName=[InputVideoName_short '.mp4'];
	load(['./' InputVideoName_short '/Data.mat'],'Data')

num_frames=Data.VideoFrames;
frames_rate=Data.VideoFrameRate;
video_size=Data.VideoSize;
width=video_size(1);
height=video_size(2);
factor = height/360;
video_fixation=Data.fixdata;

fixationPerFrame=cell(1,num_frames);
frame_durationMs=1000/frames_rate;

centermask=round(height/20);
beginflag=1;

for k=1:size(video_fixation,1)
    k;
	if k==1
		beginflag=1;
	elseif video_fixation(k,2)<video_fixation(k-1,2)
		beginflag=1;
	end
	
	vx=video_fixation(k,4);
	vy=video_fixation(k,5);
	if beginflag
		if (vx>(width/2+centermask)||vx<(width/2-centermask))&&(vy>(height/2+centermask)||vy<(height/2-centermask))&&(vx>0&&vx<width)&&(vy>0&&vy<height)
			fixPosition=[video_fixation(k,4);video_fixation(k,5)];
			beginflag=0;
		else
			continue
		end
	end
	
	if(vx>0&&vx<width)&&(vy>0&&vy<height)
		fixPosition=[video_fixation(k,4);video_fixation(k,5)];
	else
		continue
	end
	startFrame=ceil(video_fixation(k,2)/frame_durationMs);
    endFrame=ceil((video_fixation(k,2)+video_fixation(k,3))/frame_durationMs);
    if(startFrame==0)
		startFrame=1;
    end
    if(endFrame>num_frames)
        endFrame=num_frames;
    end
    for i=startFrame:endFrame
        fixationPerFrame{i}=[fixationPerFrame{i} fixPosition];
    end
end
black = 'black.avi';

%Own code
% bobj = VideoWriter(['./' InputVideoName_short '/bblack.avi']);
% bobj.FrameRate = frames_rate;
% open(bobj);
% for i=1:num_frames
%     image=zeros(360,640,3); %initialize
%     image(:,:,1)=0.0;
%     image(:,:,2)=0.0;
%     image(:,:,3)=0.0;
%     blackframe = im2frame(image);
%     writeVideo(bobj,blackframe);
% end
% close(bobj)
%Until here

obj=VideoReader(['./' InputVideoName_short '/' black]);
vidFrames=read(obj);

color_map=colormap(gray(256));

% myObj = VideoWriter(['./' InputVideoName_short '/heatmap_FV2.avi']);
% myObj.FrameRate = frames_rate;
% open(myObj);
rmdir([InputVideoName_short '/GT/map/' ], 's');
mkdir([InputVideoName_short '/GT/map/' ]);
rmdir([InputVideoName_short '/GT/fixation/' ], 's');
mkdir([InputVideoName_short '/GT/fixation/' ]);
 for k=1:num_frames
	tempframe=vidFrames(:,:,:,k);
    tempframe = imresize(tempframe,0.5);
    % mov(k).cdata = vidFrames(:,:,:,k);
    % mov(k).colormap = [];
	if ~isempty(fixationPerFrame{k})
	x=fixationPerFrame{k}(1,:);
	y=fixationPerFrame{k}(2,:);
    fixmaptemp = make_gauss_masks4(int16(x/factor),int16(y/factor),[360 640],100);
    fixationsmap = make_9mask(int16(y/factor),int16(x/factor),360, 640);
    fixmaptemp = fixmaptemp./max(fixmaptemp(:));

	[heatposrow,heatposcol]=find(fixmaptemp>0.4);
	%mov2(k).cdata = mov(k).cdata;
	for i=1:length(heatposrow)
	map_index=round(fixmaptemp(heatposrow(i),heatposcol(i))*255)+1;
	tempframe(heatposrow(i),heatposcol(i),:)=round(color_map(map_index,:)*255);
    end
    %grayImage = uint8(255) - rgb2gray(tempframe);
    
    %tempframe = cat(3, grayImage, grayImage, grayImage);
	end
    % mov(k).colormap = [];
	% salmap=mov(k).cdata;
% 	 writeVideo(myObj,tempframe);
    
    imwrite(tempframe,strcat('./', InputVideoName_short,'/GT/map/', int2str(k), '.png'));
    imwrite(fixationsmap,strcat('./', InputVideoName_short,'/GT/fixation/', int2str(k), '.png'));
 end
% close(myObj);
% close(obj);
m
toc
end