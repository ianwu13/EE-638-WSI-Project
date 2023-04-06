#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "slurm_getters" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "slurm_getters" DIRECTORY FOR THIS PROJECT'
    return -1
fi


echo "6g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_72.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_72.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_73.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_73.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_74.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_74.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_75.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_75.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_76.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_76.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_77.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_77.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_78.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_78.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_79.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_79.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_80.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_80.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_81.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_81.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_82.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_82.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_83.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_83.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_84.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_84.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_85.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_85.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_86.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_86.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_87.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_87.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_88.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_88.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_89.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_89.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_90.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_90.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_91.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_91.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_92.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_92.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_93.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_93.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_94.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_94.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/training/tumor/tumor_95.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/training/tumor/tumor_95.tif"

echo "COMPLETE"
