

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

    remote_dti1000_path =\
        "%s/derivatives/dti1000" % (outbucket)

    remote_dti1000_2000_path =\
        "%s/derivatives/dti1000_2000" % (outbucket)

    remote_dki1000_2000_path =\
        "%s/derivatives/dki1000_2000" % (outbucket)

    remote_dki2000_3000_path =\
        "%s/derivatives/dki2000_3000" % (outbucket)

    remote_dki1000_3000_path =\
        "%s/derivatives/dki1000_3000" % (outbucket)


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

    ### DTI 1000
    last_result = op.join(
        remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
        f'sub-{subject}_dwi_model-DTI_MD.nii.gz')
    if not fs.exists(last_result):
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
            dti_params = nib.load(lpath).get_fdata()
            dtif = dti.TensorFit(dtim, dti_params)

        lpath = "dti1000_fa.nii.gz"
        nib.save(nib.Nifti1Image(dtif.fa, dwi_img.affine), lpath)
        rpath = op.join(remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
                        f'sub-{subject}_dwi_model-DTI_FA.nii.gz')
        fs.upload(lpath, rpath)

        lpath = "dti1000_md.nii.gz"
        nib.save(nib.Nifti1Image(dtif.md, dwi_img.affine), lpath)
        rpath = op.join(remote_dti1000_path, f'sub-{subject}', 'ses-01', 'dwi',
                        f'sub-{subject}_dwi_model-DTI_MD.nii.gz')
        fs.upload(lpath, rpath)


    ### DTI 1000 + 2000
    last_result = op.join(
        remote_dti1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
        f'sub-{subject}_dwi_model-DTI_MD.nii.gz')
    if not fs.exists(last_result):
        lpath = "dti1000_2000_params.nii.gz"
        rpath = op.join(
            remote_dti1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DTI_diffmodel.nii.gz')

        dwi1000_2000 = dwi_data[..., gtab.bvals < 2100]
        gtab1000_2000 = gradient_table(
            gtab.bvals[gtab.bvals < 2100],
            gtab.bvecs[gtab.bvals < 2100])
        if not fs.exists(rpath):
            log.info("Fitting DTI with b=1000 and 2000")
            t1 = time.time()
            dtim = dti.TensorModel(gtab1000_2000)
            dtif = dtim.fit(dwi1000_2000, mask=np.ones(dwi_data.shape[:3]))
            nib.save(nib.Nifti1Image(dtif.model_params, dwi_img.affine), lpath)
            fs.upload(lpath, rpath)
            log.info(f"That took {time.time() - t1} seconds")
        else:
            log.info("Looks like I've already fit DTI with b=1000 and b=2000")
            log.info("Downloading DTI params from S3")
            fs.download(rpath, lpath)
            dtim = dti.TensorModel(gtab1000_2000)
            dti_params = nib.load(lpath).get_fdata()
            dtif = dti.TensorFit(dtim, dti_params)

        lpath = "dti1000_2000_fa.nii.gz"
        nib.save(nib.Nifti1Image(dtif.fa, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dti1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DTI_FA.nii.gz')
        fs.upload(lpath, rpath)

        lpath = "dti1000_2000_md.nii.gz"
        nib.save(nib.Nifti1Image(dtif.md, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dti1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DTI_MD.nii.gz')
        fs.upload(lpath, rpath)


    ### DKI 1000 + 2000
    last_result = op.join(
        remote_dki1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
        f'sub-{subject}_dwi_model-DKI_MD.nii.gz')

    if not fs.exists(last_result):
        lpath = "dki1000_2000_params.nii.gz"
        rpath = op.join(
            remote_dki1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_diffmodel.nii.gz')

        dwi1000_2000 = dwi_data[..., gtab.bvals < 2100]
        gtab1000_2000 = gradient_table(gtab.bvals[gtab.bvals < 2100],
                                       gtab.bvecs[gtab.bvals < 2100])
        if not fs.exists(rpath):
            log.info("Fitting DKI with b=1000 + 2000")
            t1 = time.time()
            dkim = dki.DiffusionKurtosisModel(gtab1000_2000)
            dkif = dkim.fit(dwi1000_2000, mask=np.ones(dwi_data.shape[:3]))
            nib.save(nib.Nifti1Image(dkif.model_params, dwi_img.affine), lpath)
            fs.upload(lpath, rpath)
            log.info(f"That took {time.time() - t1} seconds")
        else:
            log.info("Looks like I've already fit DKI with b=1000 and b=2000")
            log.info("Downloading DKI params from S3")
            fs.download(rpath, lpath)
            dkim = dki.DiffusionKurtosisModel(gtab1000_2000)
            dki_params = nib.load(lpath).get_fdata()
            dkif = dki.DiffusionKurtosisFit(dkim, dki_params)

        lpath = "dki1000_2000_fa.nii.gz"
        nib.save(nib.Nifti1Image(dkif.fa, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dki1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_FA.nii.gz')
        fs.upload(lpath, rpath)

        lpath = "dki1000_2000_md.nii.gz"
        nib.save(nib.Nifti1Image(dkif.md, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dki1000_2000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_MD.nii.gz')
        fs.upload(lpath, rpath)

    ### DKI 2000 + 3000
    last_result = op.join(
        remote_dki2000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
        f'sub-{subject}_dwi_model-DKI_MD.nii.gz')
    if not fs.exists(last_result):
        lpath = "dki2000_3000_params.nii.gz"
        rpath = op.join(
            remote_dki2000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_diffmodel.nii.gz')

        dwi2000_3000 = dwi_data[..., (gtab.bvals > 1985) | (gtab.bvals < 50)]
        gtab2000_3000 = gradient_table(
            gtab.bvals[(gtab.bvals > 1985) | (gtab.bvals < 50)],
            gtab.bvecs[(gtab.bvals > 1985) | (gtab.bvals < 50)])

        if not fs.exists(rpath):
            log.info("Fitting DKI with b=2000 + 3000")
            t1 = time.time()
            dkim = dki.DiffusionKurtosisModel(gtab2000_3000)
            dkif = dkim.fit(dwi2000_3000, mask=np.ones(dwi_data.shape[:3]))
            nib.save(nib.Nifti1Image(dkif.model_params, dwi_img.affine), lpath)
            fs.upload(lpath, rpath)
            log.info(f"That took {time.time() - t1} seconds")
        else:
            log.info("Looks like I've already fit DKI with b=2000 and b=3000")
            log.info("Downloading DKI params from S3")
            fs.download(rpath, lpath)
            dkim = dki.DiffusionKurtosisModel(gtab2000_3000)
            dki_params = nib.load(lpath).get_fdata()
            dkif = dki.DiffusionKurtosisFit(dkim, dki_params)

        lpath = "dki2000_3000_fa.nii.gz"
        nib.save(nib.Nifti1Image(dkif.fa, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dki2000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_FA.nii.gz')
        fs.upload(lpath, rpath)

        lpath = "dki2000_3000_md.nii.gz"
        nib.save(nib.Nifti1Image(dkif.md, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dki2000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_MD.nii.gz')
        fs.upload(lpath, rpath)

    ### DKI 1000 + 3000
    last_result = op.join(
        remote_dki1000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
        f'sub-{subject}_dwi_model-DKI_MD.nii.gz')
    if not fs.exists(last_result):
        lpath = "dki1000_3000_params.nii.gz"
        rpath = op.join(
            remote_dki1000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_diffmodel.nii.gz')

        dwi1000_3000 = dwi_data[..., (gtab.bvals > 2500) | (gtab.bvals < 1500)]
        gtab1000_3000 = gradient_table(
            gtab.bvals[(gtab.bvals > 2500) | (gtab.bvals < 1500)],
            gtab.bvecs[(gtab.bvals > 2500) | (gtab.bvals < 1500)])

        if not fs.exists(rpath):
            log.info("Fitting DKI with b=1000 + 3000")
            t1 = time.time()
            dkim = dki.DiffusionKurtosisModel(gtab1000_3000)
            dkif = dkim.fit(dwi1000_3000, mask=np.ones(dwi_data.shape[:3]))
            nib.save(nib.Nifti1Image(dkif.model_params, dwi_img.affine), lpath)
            fs.upload(lpath, rpath)
            log.info(f"That took {time.time() - t1} seconds")
        else:
            log.info("Looks like I've already fit DKI with b=1000 and b=3000")
            log.info("Downloading DKI params from S3")
            fs.download(rpath, lpath)
            dkim = dki.DiffusionKurtosisModel(gtab1000_3000)
            dki_params = nib.load(lpath).get_fdata()
            dkif = dki.DiffusionKurtosisFit(dkim, dki_params)

        lpath = "dki1000_3000_fa.nii.gz"
        nib.save(nib.Nifti1Image(dkif.fa, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dki1000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_FA.nii.gz')
        fs.upload(lpath, rpath)

        lpath = "dki1000_3000_md.nii.gz"
        nib.save(nib.Nifti1Image(dkif.md, dwi_img.affine), lpath)
        rpath = op.join(
            remote_dki1000_3000_path, f'sub-{subject}', 'ses-01', 'dwi',
            f'sub-{subject}_dwi_model-DKI_MD.nii.gz')
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
