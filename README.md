# MINC version of WMHsynthseg


# Original readme for WMHsynthseg is below:
# mri_WMHsynthseg

This README documents how to download and install the atlas file required by the
mri_WMHsynthseg utility. General usage and utility description can be found at:
    
    https://surfer.nmr.mgh.harvard.edu/fswiki/WMH-SynthSeg

This utility requires an atlas file that does not ship with the standard install
of FreeSurfer. This is done in an effort to keep the size of the installer
reasonable. The utility expects the atlas file, 'WMH-SynthSeg_v10_231110.pth',  
to be installed under $FREESURFER_HOME/models, and can be downloaded from an ftp
server.

Downloading the atlas:
    Linux:
        wget https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth 

    MacOS:
        curl -o WMH-SynthSeg_v10_231110.pth https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth 

Installing the atlas (same for both Linux and MacOS):
    cp WMH-SynthSeg/WMH-SynthSeg_v10_231110.pth $FREESURFER_HOME/models

You should now see 'WMH-SynthSeg_v10_231110.pth' under $FREESURFER_HOME/models
This can be confirmed, by running:
    ls $FREESURFER_HOME/models | grep WMH-SynthSeg_v10_231110.pth | wc -l

The above command should print '1' to the terminal if the model is in the proper
location.