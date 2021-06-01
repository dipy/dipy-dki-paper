from __future__ import print_function
import datetime
import io
import os
import os.path as op
import sys
import time
import configparser
try:
    input = raw_input
except NameError:
    pass

import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as batchmodels

import secrets as config
from importlib import reload
reload(config)


sys.path.append('.')
sys.path.append('..')


def wrap_commands_in_shell(ostype, commands):
    """Wrap commands in a shell

    :param list commands: list of commands to wrap
    :param str ostype: OS type, linux or windows
    :rtype: str
    :return: a shell wrapping commands
    """
    if ostype.lower() == 'linux':
        return '/bin/bash -c \'set -e; set -o pipefail; {}; wait\''.format(
            ';'.join(commands))
    elif ostype.lower() == 'windows':
        return 'cmd.exe /c "{}"'.format('&'.join(commands))
    else:
        raise ValueError('unknown ostype: {}'.format(ostype))


def query_yes_no(question, default="yes"):
    """
    Prompts the user for yes/no input, displaying the specified question text.

    :param str question: The text of the prompt for input.
    :param str default: The default if the user hits <ENTER>. Acceptable values
    are 'yes', 'no', and None.
    :rtype: str
    :return: 'yes' or 'no'
    """
    valid = {'y': 'yes', 'n': 'no'}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("Invalid default answer: '{}'".format(default))

    while 1:
        choice = input(question + prompt).lower()
        if default and not choice:
            return default
        try:
            return valid[choice[0]]
        except (KeyError, IndexError):
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def print_batch_exception(batch_exception):
    """
    Prints the contents of the specified Batch exception.

    :param batch_exception:
    """
    print('-------------------------------------------')
    print('Exception encountered:')
    if batch_exception.error and \
            batch_exception.error.message and \
            batch_exception.error.message.value:
        print(batch_exception.error.message.value)
        if batch_exception.error.values:
            print()
            for mesg in batch_exception.error.values:
                print('{}:\t{}'.format(mesg.key, mesg.value))
    print('-------------------------------------------')


def upload_file_to_container(block_blob_client, container_name, file_path):
    """
    Uploads a local file to an Azure Blob storage container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param str file_path: The local path to the file.
    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
    tasks.
    """
    blob_name = os.path.basename(file_path)

    print('Uploading file {} to container [{}]...'.format(file_path,
                                                          container_name))

    block_blob_client.create_blob_from_path(container_name,
                                            blob_name,
                                            file_path)

    sas_token = block_blob_client.generate_blob_shared_access_signature(
        container_name,
        blob_name,
        permission=azureblob.BlobPermissions.READ,
        expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2))

    sas_url = block_blob_client.make_blob_url(container_name,
                                              blob_name,
                                              sas_token=sas_token)

    return batchmodels.ResourceFile(http_url=sas_url, file_path=blob_name)


def get_container_sas_token(block_blob_client,
                            container_name, blob_permissions):
    """
    Obtains a shared access signature granting the specified permissions to the
    container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS token granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container, setting the expiry time and
    # permissions. In this case, no start time is specified, so the shared
    # access signature becomes valid immediately.
    container_sas_token = \
        block_blob_client.generate_container_shared_access_signature(
            container_name,
            permission=blob_permissions,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2))

    return container_sas_token


def create_pool(batch_service_client, pool_id, resource_files):
    """
    Creates a pool of compute nodes with the specified OS settings.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str pool_id: An ID for the new pool.
    :param str publisher: Marketplace image publisher
    :param str offer: Marketplace image offer
    :param str sku: Marketplace image sku
    """
    print('Creating pool [{}]...'.format(pool_id))

    # Create a new pool of Linux compute nodes using an Azure Virtual Machines
    # Marketplace image. For more information about creating pools of Linux
    # nodes, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/

    task_commands = [
        'cp -p {} $AZ_BATCH_NODE_SHARED_DIR'.format(config._TASK_FILE),
        '''wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
        bash miniconda.sh -b -p $AZ_BATCH_NODE_SHARED_DIR/miniconda && \
        export PATH="$AZ_BATCH_NODE_SHARED_DIR/miniconda/bin:$PATH" && \
        source "$AZ_BATCH_NODE_SHARED_DIR/miniconda/bin/activate" && \
        conda install -y -c anaconda -c conda-forge \
            pip'''
    ]
    image_ref = batchmodels.ImageReference(
                publisher="Canonical",
                offer="UbuntuServer",
                sku="18.04-LTS",
                version="latest"
            )
    vm_config = batchmodels.VirtualMachineConfiguration(
            image_reference=image_ref,
            node_agent_sku_id="batch.node.ubuntu 18.04")

    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        virtual_machine_configuration=vm_config,
        vm_size=config._POOL_VM_SIZE,
        target_low_priority_nodes=config._POOL_NODE_COUNT,
        start_task=batch.models.StartTask(
            command_line=wrap_commands_in_shell(
                            'linux',
                            task_commands),
            resource_files=resource_files
                            )
    )
    batch_service_client.pool.add(new_pool)


def create_job(batch_service_client, job_id, pool_id):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print('Creating job [{}]...'.format(job_id))

    job = batch.models.JobAddParameter(
        id=job_id,
        pool_info=batch.models.PoolInformation(pool_id=pool_id))

    batch_service_client.job.add(job)


def add_tasks(batch_service_client, job_id, subject_ids, aws_access_key,
              aws_secret_key, hcp_aws_access_key, hcp_aws_secret_key,
              outbucket):
    """
    Adds a task for each set of parameters.
    """
    print('Adding {} tasks to job [{}]...'.format(len(subject_ids), job_id))

    tasks = list()

    for idx, subject_id in enumerate(subject_ids):

        command = [
            'sleep 180 &&'  # We need to sleep here, so that all the content has time to show up in the shared dir (no kidding)
            'export PATH="$AZ_BATCH_NODE_SHARED_DIR/miniconda/bin/:$PATH" && '
            'python -m pip install sklearn &&'
            'python -m pip install git+https://github.com/yeatmanlab/pyAFQ.git@master && '
            'python $AZ_BATCH_NODE_SHARED_DIR/{} '
            '--subject {} --ak {} --sk {} '
            '--hcpak {} --hcpsk {} --outbucket {}'.format(
                config._TASK_FILE,
                subject_id,
                aws_access_key,
                aws_secret_key,
                hcp_aws_access_key,
                hcp_aws_secret_key,
                outbucket)
            ]

        tasks.append(
            batch.models.TaskAddParameter(
                id='Task{}'.format(idx),
                command_line=wrap_commands_in_shell('linux', command)
                )
            )

    batch_service_client.task.add_collection(job_id, tasks)


def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
    """
    Returns when all tasks in the specified job reach the Completed state.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The id of the job whose tasks should be to monitored.
    :param timedelta timeout: The duration to wait for task completion. If all
    tasks in the specified job do not reach Completed state within this time
    period, an exception will be raised.
    """
    timeout_expiration = datetime.datetime.now() + timeout

    print("Monitoring all tasks for 'Completed' state, timeout in {}..."
          .format(timeout), end='')

    while datetime.datetime.now() < timeout_expiration:
        print('.', end='')
        sys.stdout.flush()
        tasks = batch_service_client.task.list(job_id)

        incomplete_tasks = [task for task in tasks if
                            task.state != batchmodels.TaskState.completed]
        if not incomplete_tasks:
            print()
            return True
        else:
            time.sleep(1)

    print()
    raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within "
                       "timeout period of " + str(timeout))


def print_task_output(batch_service_client, job_id, encoding=None):
    """Prints the stdout.txt file for each task in the job.

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param str job_id: The id of the job with task output files to print.
    """

    print('Printing task output...')

    tasks = batch_service_client.task.list(job_id)

    for task in tasks:

        node_id = batch_service_client.task.get(
            job_id, task.id).node_info.node_id
        print("Task: {}".format(task.id))
        print("Node: {}".format(node_id))

        stream = batch_service_client.file.get_from_task(
            job_id, task.id, config._STANDARD_OUT_FILE_NAME)

        file_text = _read_stream_as_string(
            stream,
            encoding)
        print("stdout:")
        print(file_text)

        stream = batch_service_client.file.get_from_task(
            job_id, task.id, config._STANDARD_ERR_FILE_NAME)

        file_text = _read_stream_as_string(
            stream,
            encoding)
        print("stderror:")
        print(file_text)


def _read_stream_as_string(stream, encoding):
    """Read stream as string

    :param stream: input stream generator
    :param str encoding: The encoding of the file. The default is utf-8.
    :return: The file content.
    :rtype: str
    """
    output = io.BytesIO()
    try:
        for data in stream:
            output.write(data)
        if encoding is None:
            encoding = 'utf-8'
        return output.getvalue().decode(encoding)
    finally:
        output.close()
    raise RuntimeError('could not write data to stream or decode bytes')


if __name__ == '__main__':

    start_time = datetime.datetime.now().replace(microsecond=0)
    print('Sample start: {}'.format(start_time))
    print()

    # Create the blob client, for use in obtaining references to
    # blob storage containers and uploading files to containers.

    blob_client = azureblob.BlockBlobService(
        account_name=config._STORAGE_ACCOUNT_NAME,
        account_key=config._STORAGE_ACCOUNT_KEY)

    # # Use the blob client to create the containers in Azure Storage if they
    # # don't yet exist.

    # input_container_name = 'input'
    # blob_client.create_container(input_container_name, fail_on_exist=False)



    # # The collection of data files that are to be processed by the tasks.
    # input_file_paths = [os.path.join(sys.path[0], 'taskdata0.txt'),
    #                     os.path.join(sys.path[0], 'taskdata1.txt'),
    #                     os.path.join(sys.path[0], 'taskdata2.txt')]

    # # Upload the data files.
    # input_files = [
    #     upload_file_to_container(blob_client, input_container_name, file_path)
    #     for file_path in input_file_paths]

    app_container_name = 'afq'

    blob_client.create_container(app_container_name, fail_on_exist=False)

    application_file_paths = [op.realpath(config._TASK_FILE)]

    application_files = [
        upload_file_to_container(blob_client, app_container_name, file_path)
        for file_path in application_file_paths]

    CP = configparser.ConfigParser()
    CP.read_file(open(op.join(op.expanduser('~'), '.aws', 'credentials')))
    hcp_aws_access_key = CP.get('hcp', 'AWS_ACCESS_KEY_ID')
    hcp_aws_secret_key = CP.get('hcp', 'AWS_SECRET_ACCESS_KEY')
    aws_access_key = CP.get('default', 'AWS_ACCESS_KEY_ID')
    aws_secret_key = CP.get('default', 'AWS_SECRET_ACCESS_KEY')


    # Create a Batch service client. We'll now be interacting with the Batch
    # service in addition to Storage
    credentials = batch_auth.SharedKeyCredentials(config._BATCH_ACCOUNT_NAME,
                                                  config._BATCH_ACCOUNT_KEY)

    batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=config._BATCH_ACCOUNT_URL)

    subject_ids = config.SUBJECTS

    try:
        # Create the pool that will contain the compute nodes that will execute
        # the tasks.
        create_pool(batch_client,
                    config._POOL_ID,
                    application_files)

        # Create the job that will run the tasks.
        create_job(batch_client, config._JOB_ID, config._POOL_ID)

        # Add the tasks to the job:
        add_tasks(batch_client, config._JOB_ID, subject_ids,
                  aws_access_key, aws_secret_key, hcp_aws_access_key,
                  hcp_aws_secret_key, config._OUTBUCKET)

        # Pause execution until tasks reach Completed state.
        wait_for_tasks_to_complete(batch_client,
                                   config._JOB_ID,
                                   datetime.timedelta(minutes=config._TIMEOUT))

        print("  Success! All tasks reached the 'Completed' state within the "
              "specified timeout period.")

        # Print the stdout.txt and stderr.txt files for each task to the
        # console:
        print_task_output(batch_client, config._JOB_ID)

    except batchmodels.BatchErrorException as err:
        print_batch_exception(err)
        raise

    # Clean up storage resources
    print('Deleting container [{}]...'.format(app_container_name))
    blob_client.delete_container(app_container_name)

    # Print out some timing info
    end_time = datetime.datetime.now().replace(microsecond=0)
    print()
    print('Sample end: {}'.format(end_time))
    print('Elapsed time: {}'.format(end_time - start_time))
    print()

    # Clean up Batch resources (if the user so chooses).
    # if query_yes_no('Delete job?') == 'yes':
    batch_client.job.delete(config._JOB_ID)

    # if query_yes_no('Delete pool?') == 'yes':
    batch_client.pool.delete(config._POOL_ID)

    # # print()
    # input('Press ENTER to exit...')
