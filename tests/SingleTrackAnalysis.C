#include <TFile.h>
#include <TCanvas.h>
#include <TProfile.h>

void SingleTrackAnalysis(){
    // TFile * f = new TFile("packet-0050018-2024_07_11_19_59_52_CDT.FLOW_evt1760.hdf5.root", "READ");
    TFile * f3244 = new TFile("packet-0050018-2024_07_11_19_59_52_CDT.FLOW_evt3244_track3.hdf5.root", "READ");
    TFile * f_mip = new TFile("packet-0050017-2024_07_08_15_13_35_CDT.FLOW.rock_mu.h5.root", "READ");
    TTree * hits_3244 = (TTree*)f3244->Get("trk43/hits");
    TTree * hits_mip = (TTree*)f_mip->Get("trk33/hits");

    TProfile * p3244 = new TProfile("p3244", "p3244", 80, -4, 4);
    TProfile * pmip = new TProfile("pmip", "pmip", 80, -4, 4);

    hits_3244->Draw("Q/tinterval:dx>>p3244", "n>1", "prof");
    hits_mip->Draw("Q/tinterval:dx>>pmip", "n>1", "prof");


    p3244->SetLineColor(kRed);
    pmip->SetLineColor(kBlue);

    TF1 * landau1 = new TF1("landau1", "landau", -4, 4);
    TF1 * landau2 = new TF1("landau2", "landau", -4, 4);

    landau1->SetLineColor(kRed);
    landau2->SetLineColor(kBlue);

    pmip->Fit(landau2);
    p3244->Fit(landau1);
    

    TCanvas * c = new TCanvas("c", "c", 800, 600);
    p3244->Draw();
    pmip->Draw("same");

    TLegend * leg = new TLegend(0.1, 0.7, 0.3, 0.9);
    leg->AddEntry(p3244, "Single Track", "l");
    leg->AddEntry(pmip, "MIP", "l");

    leg->Draw();
}