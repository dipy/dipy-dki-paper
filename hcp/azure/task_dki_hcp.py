

import argparse
import logging
from AFQ.data import fetch_hcp
import s3fs
import nibabel as nib
from dipy.core.gradients import gradient_table
import os.path as op
import AFQ.data as afd
from dipy.reconst import dti, dki
import time
import numpy as np


def hcp_dki(subject, aws_access_key, aws_secret_key, hcp_aws_access_key,
            hcp_aws_secret_key, outbucket):

    fs = s3fs.S3FileSystem(key=aws_access_key, secret=aws_secret_key)

    remote_dti_path =\
        "%s/derivatives/dti" % (outbucket)

    remote_dti1000_path =\
        "%s/derivatives/dti1000" % (outbucket)

    remote_dki_path =\
        "%s/derivatives/dki" % (outbucket)

    remote_sst_path =\
        "%s/derivatives/sst" % (outbucket)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__) # noqa

    log.info(f"Getting data for subject {subject}")
    # get HCP data for the given subject / session
    _, hcp_bids = fetch_hcp(
        [subject],
        profile_name=False,
        aws_access_key_id=hcp_aws_access_key,
        aws_secret_access_key=hcp_aws_secret_key)

    dwi_path = op.join(afd.afq_home, 'HCP_1200', 'derivatives', 'dmriprep',
                       f'sub-{subject}', 'ses-01', 'dwi')

    dwi_img = nib.load(op.join(dwi_path, f'sub-{subject}_dwi.nii.gz'))
    dwi_data = dwi_img.get_fdata()
    b0_threshold = 50

    gtab = gradient_table(
        op.join(dwi_path, f'sub-{subject}_dwi.bval'),
        op.join(dwi_path, f'sub-{subject}_dwi.bvec'),
        b0_threshold=b0_threshold)

    log.info("Calculating SST")
    data_dwi = dwi_data[..., ~gtab.b0s_mask]
    mean_dwi = np.mean(dwi_data[..., ~gtab.b0s_mask], -1)
    sst = np.sum((data_dwi - mean_dwi[..., None]) ** 2, -1)
    lpath = "data_sst.nii.gz"
    nib.save(nib.Nifti1Image(sst, dwi_img.affine), lpath)
    rpath = op.join(remote_sst_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_desc-sst.nii.gz')
    fs.upload(lpath, rpath)

    lpath = "dti_params.nii.gz"
    rpath = op.join(remote_dti_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_model-DTI_diffmodel.nii.gz')

    if not fs.exists(rpath):
        log.info("Fitting DTI")
        t1 = time.time()
        dtim = dti.TensorModel(gtab)
        dtif = dtim.fit(dwi_data, mask=np.ones(dwi_data.shape[:3]))
        nib.save(nib.Nifti1Image(dtif.model_params, dwi_img.affine), lpath)
        fs.upload(lpath, rpath)
        log.info(f"That took {time.time() - t1} seconds")
    else:
        log.info("Looks like I've already fit DTI")
        log.info("Downloading DTI params from S3")
        fs.download(rpath, lpath)
        dtim = dti.TensorModel(gtab)

    dti_params = nib.load("dti_params.nii.gz")
    S0 = np.mean(dwi_data[..., gtab.b0s_mask], -1)
    pred = dtim.predict(dti_params.get_fdata(), S0=S0)

    lpath = "dti_pred.nii.gz"
    rpath = op.join(remote_dti_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_prediction-DTI_diffmodel.nii.gz')

    nib.save(nib.Nifti1Image(pred, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)

    # We calculate SSE only over diffusion-weighted volumes
    sse = np.sum((pred[..., ~gtab.b0s_mask] -
                  dwi_data[..., ~gtab.b0s_mask]) ** 2, -1)
    lpath = "dti_sse.nii.gz"
    rpath = op.join(remote_dti_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_SSE-DTI_diffmodel.nii.gz')
    nib.save(nib.Nifti1Image(sse, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)

    ### DTI 1000
    lpath = "dti1000_params.nii.gz"
    rpath = op.join(remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_model-DTI_diffmodel.nii.gz')

    dwi1000 = dwi_data[..., gtab.bvals < 1100]
    gtab1000 = gradient_table(gtab.bvals[gtab.bvals < 1100],
                              gtab.bvecs[gtab.bvals < 1100])
    if not fs.exists(rpath):
        log.info("Fitting DTI")
        t1 = time.time()
        dtim = dti.TensorModel(gtab1000)
        dtif = dtim.fit(dwi1000, mask=np.ones(dwi_data.shape[:3]))
        nib.save(nib.Nifti1Image(dtif.model_params, dwi_img.affine), lpath)
        fs.upload(lpath, rpath)
        log.info(f"That took {time.time() - t1} seconds")
    else:
        log.info("Looks like I've already fit DTI with b=1000")
        log.info("Downloading DTI params from S3")
        fs.download(rpath, lpath)
        dtim = dti.TensorModel(gtab1000)

    dti_params = nib.load("dti_params.nii.gz")
    S0 = np.mean(dwi1000[..., gtab1000.b0s_mask], -1)
    pred = dtim.predict(dti_params.get_fdata(), S0=S0)

    lpath = "dti1000_pred.nii.gz"
    rpath = op.join(remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_prediction-DTI_diffmodel.nii.gz')

    nib.save(nib.Nifti1Image(pred, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)

    # We calculate SSE only over diffusion-weighted volumes
    sse = np.sum((pred[..., ~gtab1000.b0s_mask] -
                  dwi1000[..., ~gtab1000.b0s_mask]) ** 2, -1)
    lpath = "dti1000_sse.nii.gz"
    rpath = op.join(remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_SSE-DTI_diffmodel.nii.gz')
    nib.save(nib.Nifti1Image(sse, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)

    ### DKI
    lpath = "dki_params.nii.gz"
    rpath = op.join(remote_dki_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_model-DKI_diffmodel.nii.gz')

    if not fs.exists(rpath):
        log.info("Fitting DKI")
        t1 = time.time()
        dkim = dki.DiffusionKurtosisModel(gtab)
        dkif = dkim.fit(dwi_data)
        log.info(f"That took {time.time() - t1} seconds")
        nib.save(nib.Nifti1Image(dkif.model_params, dwi_img.affine), lpath)
        fs.upload(lpath, rpath)
    else:
        log.info("Looks like I've already fit DKI")
        log.info("Downloading DKI params from S3")
        fs.download(rpath, lpath)
        dkim = dki.DiffusionKurtosisModel(gtab)

    dki_params = nib.load("dki_params.nii.gz")
    pred = dkim.predict(dki_params.get_fdata(), S0=S0)
    lpath = "dki_pred.nii.gz"
    rpath = op.join(remote_dki_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_prediction-DKI_diffmodel.nii.gz')

    nib.save(nib.Nifti1Image(pred, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)

    # We calculate SSE only over diffusion-weighted volumes
    sse = np.sum((pred[..., ~gtab.b0s_mask] -
                  dwi_data[..., ~gtab.b0s_mask]) ** 2, -1)
    lpath = "dki_sse.nii.gz"
    rpath = op.join(remote_dki_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_SSE-DKI_diffmodel.nii.gz')

    nib.save(nib.Nifti1Image(sse, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)

    sse1000 = np.sum((pred[..., gtab.bvals < 1100] -
                      dwi_data[..., gtab.bvals < 1100]) ** 2, -1)
    lpath = "dki1000_sse.nii.gz"
    rpath = op.join(remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
                    f'sub-{subject}_dwi_SSE-DKI_diffmodel.nii.gz')
    nib.save(nib.Nifti1Image(sse1000, dwi_img.affine), lpath)
    fs.upload(lpath, rpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, required=True,
                        help='subject ID in the HCP dataset')
    parser.add_argument('--ak', type=str, required=True,
                        help='AWS Access Key')
    parser.add_argument('--sk', type=str, required=True,
                        help='AWS Secret Key')
    parser.add_argument('--hcpak', type=str, required=True,
                        help='AWS Access Key for HCP dataset')
    parser.add_argument('--hcpsk', type=str, required=True,
                        help='AWS Secret Key for HCP dataset')
    parser.add_argument('--outbucket', type=str, required=True,
                        help='Where do I put the outputs')

    args = parser.parse_args()
    hcp_dki(args.subject,
            args.ak, args.sk,
            args.hcpak, args.hcpsk,
            args.outbucket)
