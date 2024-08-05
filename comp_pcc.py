import torch
import numpy as np

#Comparison Metric
'''golden - torchvision model
calculate - reference model
'''

def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        print("Both tensors are 'nan'")
        return True, f"PCC: {1.0}"

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        print("One tensor is all nan, the other is not.")
        return False, f"PCC: {0.0}"

    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        print("One tensor is all zero")
        return False, f"PCC: {0.0}"

    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, f"PCC: {1.0}"

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, f"PCC: {1.0}"

    return cal_pcc >= pcc, f"PCC: {cal_pcc}"
