module matrix_pe(
  input                 clk,
  input                 rst_n,
  input         [511:0] nram_mpe_neuron,
  input                 nram_mpe_neuron_valid,
  output                nram_mpe_neuron_ready,
  input         [511:0] wram_mpe_weight,
  input                 wram_mpe_weight_valid,
  output                wram_mpe_weight_ready,
  input         [  7:0] ib_ctl_uop,                //inst
  input                 ib_ctl_uop_valid,
  output reg            ib_ctl_uop_ready,
  output        [ 31:0] result,
  output                vld_o
);


wire [1:0] pe_ctl;



reg inst_vld;
wire pe_vld_i = inst_vld && nram_mpe_neuron_valid && wram_mpe_weight_valid;  //数据有效


reg [7:0] inst;
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    inst <= 1'b0;
    inst_vld <= 1'b0;
  end else if(ib_ctl_uop_valid && nram_mpe_neuron_valid && wram_mpe_weight_valid) begin
    inst <= ib_ctl_uop;
    inst_vld <= 1'b1;
  end else if(pe_ctl[1] && pe_vld_i ) begin  //最后一个数据输入进来后
    inst <= 1'b0;
    inst_vld <= 1'b0;
  end
end
reg [7:0] iter;
always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    iter <= 8'b0;
  end else if(pe_ctl[1] && pe_vld_i ) begin //最后一个数据输入进来
    iter <= 8'b0;
  end else if(pe_vld_i && nram_mpe_neuron_valid && wram_mpe_weight_valid) begin
    iter <= iter + 1'b1;
  end
end

always@(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
    ib_ctl_uop_ready <= 1'b0;
  end else if(pe_ctl[1]) begin
    ib_ctl_uop_ready <= 1'b1;
  end else if(ib_ctl_uop_ready) begin
    ib_ctl_uop_ready <= 1'b0;
  end
end

wire [511:0] pe_neuron = nram_mpe_neuron;
wire [511:0] pe_weight = wram_mpe_weight;
assign pe_ctl[0] = (iter[7:0] == 8'h0) && nram_mpe_neuron_valid && wram_mpe_weight_valid ;
assign pe_ctl[1] = (iter[7:0] == (inst[7:0] - 1'b1)) && nram_mpe_neuron_valid && wram_mpe_weight_valid ;

wire [31:0] pe_result;
wire pe_vld_o;
parallel_pe u_parallel_pe (
  .clk                  (clk      ),
  .rst_n                (rst_n    ),
  .neuron               (pe_neuron),
  .weight               (pe_weight),
  .ctl                  (pe_ctl   ),
  .vld_i                (pe_vld_i ),
  .result               (pe_result),
  .vld_o                (pe_vld_o )
);

assign nram_mpe_neuron_ready = pe_vld_i; 
assign wram_mpe_weight_ready = pe_vld_i; 

assign result = pe_result;
assign vld_o = pe_vld_o;

endmodule
