module parallel_pe(
  input                 clk,
  input                 rst_n,
  input         [511:0] neuron,
  input         [511:0] weight,
  input         [  1:0] ctl,
  input                 vld_i,
  output        [ 31:0] result,
  output reg            vld_o
);

wire [1023:0] mult_result;
pe_mult u_pe_mult(
  .mult_neuron (neuron),
  .mult_weight (weight),
  .mult_result (mult_result)
);

wire [31:0] acc_result;
pe_acc u_pe_acc(
  .mult_result (mult_result),
  .acc_result  (acc_result)
);

reg [31:0] psum_r;
wire [31:0] psum_d = ctl[0] ? acc_result : acc_result + psum_r;

always@(posedge clk or negedge rst_n)
if(!rst_n) begin
  psum_r <= 32'h0;
end else if(vld_i) begin
  psum_r <= psum_d;
end

always@(posedge clk or negedge rst_n)
if(!rst_n) begin
  vld_o <= 1'b0;
end else if(ctl[1]) begin
  vld_o <= 1'b1;
end else begin
  vld_o <= 1'b0;
end

assign result = psum_r;

endmodule
